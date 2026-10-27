//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the pass which inserts x86 AVX vzeroupper instructions
/// before calls to SSE encoded functions, and before legacy SSE instructions
/// that write an XMM register, which under AVX means the ones with no VEX
/// encoding (SHA, SSE4A, KL, WIDEKL). This avoids transition latency penalty
/// when transferring control between AVX encoded instructions and old SSE
/// encoding mode.
///
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "x86-insert-vzeroupper"

STATISTIC(NumVZU, "Number of vzeroupper instructions inserted");

namespace {
class X86InsertVZeroUpperLegacy : public MachineFunctionPass {
public:
  static char ID;

  X86InsertVZeroUpperLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "X86 vzeroupper inserter"; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};

enum BlockExitState { PASS_THROUGH, EXITS_CLEAN, EXITS_DIRTY };

// Core algorithm state:
// BlockState - Each block is either:
//   - PASS_THROUGH: There are neither YMM/ZMM dirtying instructions nor
//                   vzeroupper instructions in this block.
//   - EXITS_CLEAN: There is (or will be) a vzeroupper instruction in this
//                  block that will ensure that YMM/ZMM is clean on exit.
//   - EXITS_DIRTY: An instruction in the block dirties YMM/ZMM and no
//                  subsequent vzeroupper in the block clears it.
//
// AddedToDirtySuccessors - This flag is raised when a block is added to the
//                          DirtySuccessors list to ensure that it's not
//                          added multiple times.
//
// FirstUnguardedCall - Records the location of the first unguarded call (or
//                      legacy SSE instruction) in each basic block that may
//                      need to be guarded by a vzeroupper. We won't know
//                      whether it actually needs to be guarded until we
//                      discover a predecessor that is DIRTY_OUT.
struct BlockState {
  BlockExitState ExitState = PASS_THROUGH;
  bool AddedToDirtySuccessors = false;
  MachineBasicBlock::iterator FirstUnguardedCall;

  BlockState() = default;
};

/// A YMM register live across a guarded legacy SSE instruction, which the
/// VZEROUPPER in front of it clobbers. Def loaded it from the constant pool
/// earlier in the block, and is repeated before Use, the next instruction that
/// reads more than its low 128 bits.
struct RematYmm {
  MCRegister Reg;
  MachineInstr *Def;
  MachineInstr *Use;
};

using BlockStateMap = SmallVector<BlockState, 8>;
using DirtySuccessorsWorkList = SmallVector<MachineBasicBlock *, 8>;
} // end anonymous namespace

char X86InsertVZeroUpperLegacy::ID = 0;

FunctionPass *llvm::createX86InsertVZeroUpperLegacyPass() {
  return new X86InsertVZeroUpperLegacy();
}

#ifndef NDEBUG
static const char *getBlockExitStateName(BlockExitState ST) {
  switch (ST) {
  case PASS_THROUGH:
    return "Pass-through";
  case EXITS_DIRTY:
    return "Exits-dirty";
  case EXITS_CLEAN:
    return "Exits-clean";
  }
  llvm_unreachable("Invalid block exit state.");
}
#endif

/// VZEROUPPER cleans state that is related to Y/ZMM0-15 only.
/// Thus, there is no need to check for Y/ZMM16 and above.
static bool isYmmOrZmmReg(MCRegister Reg) {
  return (Reg >= X86::YMM0 && Reg <= X86::YMM15) ||
         (Reg >= X86::ZMM0 && Reg <= X86::ZMM15);
}

/// Returns true if MI is legacy SSE encoded and writes an XMM register, which
/// pays the SSE/AVX transition penalty when the upper state is dirty. Under AVX
/// only instructions with no VEX form match, such as SHA, SSE4A, KL and WIDEKL.
/// Implicit defs count, since ENCODEKEY* and the WIDEKL instructions write
/// their XMM results only that way.
static bool isLegacySSEWritingXmm(const MachineInstr &MI) {
  // Pseudos (IMPLICIT_DEF, KILL, ...) have no encoding but read as LEGACY.
  if (MI.isPseudo() ||
      (MI.getDesc().TSFlags & X86II::EncodingMask) != X86II::LEGACY)
    return false;
  return any_of(MI.all_defs(), [](const MachineOperand &MO) {
    return X86::VR128RegClass.contains(MO.getReg());
  });
}

/// TII->isTriviallyReMaterializable() would accept most of these, but not the
/// AVX2 broadcast loads (VPBROADCAST*Yrm, VBROADCASTI128rm).
static bool isConstantPoolLoad(const MachineInstr &MI) {
  return MI.mayLoad() && !MI.mayStore() && MI.getNumExplicitDefs() == 1 &&
         MI.isDereferenceableInvariantLoad() &&
         any_of(MI.operands(),
                [](const MachineOperand &MO) { return MO.isCPI(); }) &&
         all_of(MI.uses(), [](const MachineOperand &MO) {
           return !MO.isReg() || !MO.getReg() || MO.getReg() == X86::RIP;
         });
}

static bool readsUpperYmm(const MachineInstr &MI, MCRegister Ymm,
                          const TargetRegisterInfo *TRI) {
  return any_of(MI.operands(), [&](const MachineOperand &MO) {
    return MO.isReg() && MO.readsReg() &&
           TRI->isSuperRegisterEq(Ymm, MO.getReg());
  });
}

/// Collect the legacy SSE instructions in MBB that a VZEROUPPER can go in
/// front of: those with no YMM/ZMM0-15 register live across them, or whose
/// only live YMM registers can be reloaded afterwards (recorded in Remat).
/// LiveRegUnits would not do: XMMn and YMMn share register units, so a live
/// XMMn would block the guard.
static void collectGuardableLegacySSE(
    MachineBasicBlock &MBB, SmallPtrSetImpl<MachineInstr *> &Set,
    DenseMap<MachineInstr *, SmallVector<RematYmm, 1>> &Remat) {
  const TargetRegisterInfo *TRI =
      MBB.getParent()->getSubtarget().getRegisterInfo();
  // Above -O0, LVI load hardening fences loads before this pass runs, so a
  // reload added here would go unfenced.
  bool CanReload =
      !MBB.getParent()->getSubtarget<X86Subtarget>().useLVILoadHardening();
  // Backward: liveness, and the next read of each YMM's upper half.
  std::array<MachineInstr *, 16> NextUse = {};
  MachineBasicBlock::iterator FirstTerm = MBB.getFirstTerminator();
  LivePhysRegs LPR(*TRI);
  LPR.addLiveOuts(MBB);
  for (MachineInstr &MI : reverse(MBB)) {
    if (MI.isDebugInstr())
      continue;
    LPR.stepBackward(MI);
    if (isLegacySSEWritingXmm(MI)) {
      SmallVector<RematYmm, 1> Restore;
      bool Guardable = true;
      for (MCPhysReg R : LPR) {
        if (!isYmmOrZmmReg(R))
          continue;
        if (!CanReload || !X86::VR256RegClass.contains(R) ||
            !NextUse[R - X86::YMM0]) {
          Guardable = false;
          break;
        }
        Restore.push_back({R, nullptr, NextUse[R - X86::YMM0]});
      }
      if (Guardable) {
        Set.insert(&MI);
        if (!Restore.empty())
          Remat[&MI] = std::move(Restore);
      }
    }
    // The block can be left at a terminator, a call that throws, or inline
    // asm (asm goto, or asm that unwinds), so a reload after one of them may
    // not cover every path. Anything else with unmodeled side effects is
    // treated the same way. A reload can still go right before such an
    // instruction, but not between two terminators.
    if (MI.isCall() || MI.isTerminator() || MI.isInlineAsm() ||
        MI.hasUnmodeledSideEffects())
      NextUse.fill(nullptr);
    if (MI.isTerminator() && MachineBasicBlock::iterator(MI) != FirstTerm)
      continue;
    for (unsigned I = 0; I != NextUse.size(); ++I) {
      MCRegister Ymm = X86::YMM0 + I;
      if (readsUpperYmm(MI, Ymm, TRI))
        NextUse[I] = &MI;
      else if (MI.modifiesRegister(Ymm, TRI))
        NextUse[I] = nullptr;
    }
  }
  if (Remat.empty())
    return;
  // Forward: the constant-pool load each live YMM came from, if any.
  std::array<MachineInstr *, 16> LastDef = {};
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    auto It = Remat.find(&MI);
    if (It != Remat.end()) {
      for (RematYmm &RY : It->second)
        RY.Def = LastDef[RY.Reg - X86::YMM0];
      if (any_of(It->second, [](const RematYmm &RY) { return !RY.Def; })) {
        Set.erase(&MI);
        Remat.erase(It);
      }
    }
    for (unsigned I = 0; I != LastDef.size(); ++I) {
      MCRegister Ymm = X86::YMM0 + I;
      if (MI.modifiesRegister(Ymm, TRI))
        LastDef[I] = isConstantPoolLoad(MI) && MI.getOperand(0).getReg() == Ymm
                         ? &MI
                         : nullptr;
    }
  }
}

static bool checkFnHasLiveInYmmOrZmm(MachineRegisterInfo &MRI) {
  for (std::pair<MCRegister, Register> LI : MRI.liveins())
    if (isYmmOrZmmReg(LI.first))
      return true;

  return false;
}

static bool clobbersAllYmmAndZmmRegs(const MachineOperand &MO) {
  for (unsigned reg = X86::YMM0; reg <= X86::YMM15; ++reg) {
    if (!MO.clobbersPhysReg(reg))
      return false;
  }
  for (unsigned reg = X86::ZMM0; reg <= X86::ZMM15; ++reg) {
    if (!MO.clobbersPhysReg(reg))
      return false;
  }
  return true;
}

static bool hasYmmOrZmmReg(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (MI.isCall() && MO.isRegMask() && !clobbersAllYmmAndZmmRegs(MO))
      return true;
    if (!MO.isReg())
      continue;
    if (MO.isDebug())
      continue;
    if (isYmmOrZmmReg(MO.getReg().asMCReg()))
      return true;
  }
  return false;
}

/// Check if given call instruction has a RegMask operand.
static bool callHasRegMask(MachineInstr &MI) {
  assert(MI.isCall() && "Can only be called on call instructions.");
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask())
      return true;
  }
  return false;
}

/// Insert a vzeroupper instruction before I.
static bool insertVZeroUpper(MachineBasicBlock::iterator I,
                             MachineBasicBlock &MBB,
                             const TargetInstrInfo *TII) {
  BuildMI(MBB, I, I->getDebugLoc(), TII->get(X86::VZEROUPPER));
  ++NumVZU;
  return true;
}

/// Add MBB to the DirtySuccessors list if it hasn't already been added.
static void addDirtySuccessor(MachineBasicBlock &MBB,
                              BlockStateMap &BlockStates,
                              DirtySuccessorsWorkList &DirtySuccessors) {
  if (!BlockStates[MBB.getNumber()].AddedToDirtySuccessors) {
    DirtySuccessors.push_back(&MBB);
    BlockStates[MBB.getNumber()].AddedToDirtySuccessors = true;
  }
}

/// Loop over all of the instructions in the basic block, inserting vzeroupper
/// instructions before function calls and legacy SSE instructions.
static bool processBasicBlock(MachineBasicBlock &MBB,
                              BlockStateMap &BlockStates,
                              DirtySuccessorsWorkList &DirtySuccessors,
                              bool IsX86INTR, const TargetInstrInfo *TII) {
  // Start by assuming that the block is PASS_THROUGH which implies no unguarded
  // calls.
  BlockExitState CurState = PASS_THROUGH;
  BlockStates[MBB.getNumber()].FirstUnguardedCall = MBB.end();
  bool MadeChange = false;

  // Finding guardable legacy SSE instructions needs valid block live-ins, so
  // only do it when liveness is tracked, and lazily since few blocks have any.
  bool GuardLegacySSE = MBB.getParent()->getProperties().hasTracksLiveness();
  SmallPtrSet<MachineInstr *, 8> GuardableLegacySSE;
  DenseMap<MachineInstr *, SmallVector<RematYmm, 1>> RematLegacySSE;
  SmallSet<std::pair<MachineInstr *, MachineInstr *>, 2> Rematerialized;
  bool ComputedGuardable = false;

  for (MachineInstr &MI : MBB) {
    bool IsCall = MI.isCall();
    bool IsReturn = MI.isReturn();
    bool IsControlFlow = IsCall || IsReturn;

    // No need for vzeroupper before iret in interrupt handler function,
    // epilogue will restore YMM/ZMM registers if needed.
    if (IsX86INTR && IsReturn)
      continue;

    // An existing VZERO* instruction resets the state.
    if (MI.getOpcode() == X86::VZEROALL || MI.getOpcode() == X86::VZEROUPPER) {
      CurState = EXITS_CLEAN;
      continue;
    }

    // A legacy SSE instruction needs a clean upper state just like a call
    // does, provided any YMM/ZMM value live across it can be reloaded after.
    bool NeedsCleanState = false;
    if (GuardLegacySSE && !IsControlFlow && CurState != EXITS_CLEAN &&
        isLegacySSEWritingXmm(MI)) {
      if (!ComputedGuardable) {
        collectGuardableLegacySSE(MBB, GuardableLegacySSE, RematLegacySSE);
        ComputedGuardable = true;
      }
      NeedsCleanState = GuardableLegacySSE.contains(&MI);
    }

    // Shortcut: don't need to check regular instructions in dirty state.
    if (!IsControlFlow && !NeedsCleanState && CurState == EXITS_DIRTY)
      continue;

    if (hasYmmOrZmmReg(MI)) {
      // We found a ymm/zmm-using instruction; this could be an AVX/AVX512
      // instruction, or it could be control flow.
      CurState = EXITS_DIRTY;
      continue;
    }

    // Check for control-flow out of the current function (which might
    // indirectly execute SSE instructions), or a legacy SSE instruction.
    if (!IsControlFlow && !NeedsCleanState)
      continue;

    // If the call has no RegMask, skip it as well. It usually happens on
    // helper function calls (such as '_chkstk', '_ftol2') where standard
    // calling convention is not used (RegMask is not used to mark register
    // clobbered and register usage (def/implicit-def/use) is well-defined and
    // explicitly specified.
    if (IsCall && !callHasRegMask(MI))
      continue;

    // The VZEROUPPER instruction resets the upper 128 bits of YMM0-YMM15
    // registers. In addition, the processor changes back to Clean state, after
    // which execution of SSE instructions or AVX instructions has no transition
    // penalty. Add the VZEROUPPER instruction before any function call/return
    // that might execute SSE code, and before legacy SSE instructions.
    // FIXME: In some cases, we may want to move the VZEROUPPER into a
    // predecessor block.
    if (CurState == EXITS_DIRTY) {
      // After the inserted VZEROUPPER the state becomes clean again, but
      // other YMM/ZMM may appear before other subsequent calls or even before
      // the end of the BB.
      MadeChange |= insertVZeroUpper(MI, MBB, TII);
      CurState = EXITS_CLEAN;
      // Reload any YMM constant the VZEROUPPER clobbered before its upper half
      // is next read.
      if (NeedsCleanState) {
        for (const RematYmm &RY : RematLegacySSE.lookup(&MI))
          if (Rematerialized.insert({RY.Def, RY.Use}).second)
            MBB.insert(MachineBasicBlock::iterator(RY.Use),
                       MBB.getParent()->CloneMachineInstr(RY.Def));
      }
    } else if (CurState == PASS_THROUGH) {
      // If this block is currently in pass-through state and we encounter a
      // call or legacy SSE instruction then whether we need a vzeroupper or not
      // depends on whether this block has successors that exit dirty. Record
      // its location, and set the state to EXITS_CLEAN, but do not insert the
      // vzeroupper yet. It will be inserted later if necessary.
      // A reloadable YMM was loaded earlier in this block, which left it dirty.
      assert(!RematLegacySSE.count(&MI) && "Deferred guard cannot reload YMM");
      BlockStates[MBB.getNumber()].FirstUnguardedCall = MI;
      CurState = EXITS_CLEAN;
    }
  }

  LLVM_DEBUG(dbgs() << "MBB #" << MBB.getNumber() << " exit state: "
                    << getBlockExitStateName(CurState) << '\n');

  if (CurState == EXITS_DIRTY)
    for (MachineBasicBlock *Succ : MBB.successors())
      addDirtySuccessor(*Succ, BlockStates, DirtySuccessors);

  BlockStates[MBB.getNumber()].ExitState = CurState;
  return MadeChange;
}

/// Loop over all of the basic blocks, inserting vzeroupper instructions before
/// function calls and legacy SSE instructions.
static bool insertVZeroUpper(MachineFunction &MF) {
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.getCLOpts().use_vzeroupper || !ST.hasAVX() || !ST.insertVZEROUPPER())
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();

  bool FnHasLiveInYmmOrZmm = checkFnHasLiveInYmmOrZmm(MRI);

  // Fast check: if the function doesn't use any ymm/zmm registers, we don't
  // need to insert any VZEROUPPER instructions.  This is constant-time, so it
  // is cheap in the common case of no ymm/zmm use.
  bool YmmOrZmmUsed = FnHasLiveInYmmOrZmm;
  for (const auto *RC : {&X86::VR256RegClass, &X86::VR512_0_15RegClass}) {
    if (!YmmOrZmmUsed) {
      for (MCPhysReg R : *RC) {
        if (!MRI.reg_nodbg_empty(R)) {
          YmmOrZmmUsed = true;
          break;
        }
      }
    }
  }
  if (!YmmOrZmmUsed)
    return false;

  const TargetInstrInfo *TII = ST.getInstrInfo();
  bool IsX86INTR = MF.getFunction().getCallingConv() == CallingConv::X86_INTR;
  bool EverMadeChange = false;
  BlockStateMap BlockStates(MF.getNumBlockIDs());
  DirtySuccessorsWorkList DirtySuccessors;

  assert(BlockStates.size() == MF.getNumBlockIDs() && DirtySuccessors.empty() &&
         "X86VZeroUpper state should be clear");

  // Process all blocks. This will compute block exit states, record the first
  // unguarded call in each block, and add successors of dirty blocks to the
  // DirtySuccessors list.
  for (MachineBasicBlock &MBB : MF)
    EverMadeChange |=
        processBasicBlock(MBB, BlockStates, DirtySuccessors, IsX86INTR, TII);

  // If any YMM/ZMM regs are live-in to this function, add the entry block to
  // the DirtySuccessors list
  if (FnHasLiveInYmmOrZmm)
    addDirtySuccessor(MF.front(), BlockStates, DirtySuccessors);

  // Re-visit all blocks that are successors of EXITS_DIRTY blocks. Add
  // vzeroupper instructions to unguarded calls, and propagate EXITS_DIRTY
  // through PASS_THROUGH blocks.
  while (!DirtySuccessors.empty()) {
    MachineBasicBlock &MBB = *DirtySuccessors.back();
    DirtySuccessors.pop_back();
    BlockState &BBState = BlockStates[MBB.getNumber()];

    // MBB is a successor of a dirty block, so its first call or legacy SSE
    // instruction needs to be guarded.
    if (BBState.FirstUnguardedCall != MBB.end())
      EverMadeChange |= insertVZeroUpper(BBState.FirstUnguardedCall, MBB, TII);

    // If this successor was a pass-through block, then it is now dirty. Its
    // successors need to be added to the worklist (if they haven't been
    // already).
    if (BBState.ExitState == PASS_THROUGH) {
      LLVM_DEBUG(dbgs() << "MBB #" << MBB.getNumber()
                        << " was Pass-through, is now Dirty-out.\n");
      for (MachineBasicBlock *Succ : MBB.successors())
        addDirtySuccessor(*Succ, BlockStates, DirtySuccessors);
    }
  }

  return EverMadeChange;
}

bool X86InsertVZeroUpperLegacy::runOnMachineFunction(MachineFunction &MF) {
  return insertVZeroUpper(MF);
}

PreservedAnalyses
X86InsertVZeroUpperPass::run(MachineFunction &MF,
                             MachineFunctionAnalysisManager &MFAM) {
  return insertVZeroUpper(MF) ? getMachineFunctionPassPreservedAnalyses()
                                    .preserveSet<CFGAnalyses>()
                              : PreservedAnalyses::all();
}
