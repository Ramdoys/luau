// This file is part of the Luau programming language and is licensed under MIT License; see LICENSE.txt for details
#include "EmitInstructionX64.h"

#include "Luau/AssemblyBuilderX64.h"
#include "Luau/IrRegAllocX64.h"
#include "Luau/IrCallWrapperX64.h"

#include "EmitCommonX64.h"
#include "NativeState.h"

#include "lstate.h"

namespace Luau
{
namespace CodeGen
{
namespace X64
{

void emitInstCall(IrRegAllocX64& regs, AssemblyBuilderX64& build, ModuleHelpers& helpers, int ra, int nparams, int nresults, uint32_t instIdx)
{
    IrCallWrapperX64 callPrologWrap(regs, build, instIdx);
    callPrologWrap.addArgument(SizeX64::qword, rState);
    callPrologWrap.addArgument(SizeX64::qword, luauRegAddress(ra));

    OperandX64 targetTop = (nparams == LUA_MULTRET)
                               ? OperandX64(qword[rState + offsetof(lua_State, top)])
                               : luauRegAddress(ra + 1 + nparams);
    callPrologWrap.addArgument(SizeX64::qword, targetTop);
    callPrologWrap.addArgument(SizeX64::dword, OperandX64(nresults));
    callPrologWrap.call(qword[rNativeContext + offsetof(NativeContext, callProlog)]);
    RegisterX64 ccl = rax;

    emitUpdateBase(build);

    Label cFuncCall;

    build.test(byte[ccl + offsetof(Closure, isC)], 1);
    build.jcc(ConditionX64::NotZero, cFuncCall);

    {
        RegisterX64 proto = rcx; // Sync with emitContinueCallInVm
        RegisterX64 ci = rdx;
        RegisterX64 argi = rsi;
        RegisterX64 argend = rdi;

        build.mov(proto, qword[ccl + offsetof(Closure, l.p)]);

        // Switch current Closure
        build.mov(sClosure, ccl); // Last use of 'ccl'

        build.mov(ci, qword[rState + offsetof(lua_State, ci)]);

        Label fillnil, exitfillnil;

        // argi = L->top
        build.mov(argi, qword[rState + offsetof(lua_State, top)]);

        // argend = L->base + p->numparams
        build.movzx(eax, byte[proto + offsetof(Proto, numparams)]);
        build.shl(eax, kTValueSizeLog2);
        build.lea(argend, addr[rBase + rax]);

        // while (argi < argend) setnilvalue(argi++);
        build.setLabel(fillnil);
        build.cmp(argi, argend);
        build.jcc(ConditionX64::NotBelow, exitfillnil);

        build.mov(dword[argi + offsetof(TValue, tt)], LUA_TNIL);
        build.add(argi, sizeof(TValue));
        build.jmp(fillnil); // This loop rarely runs so it's not worth repeating cmp/jcc

        build.setLabel(exitfillnil);

        // Set L->top to ci->top as most function expect (no vararg)
        build.mov(rax, qword[ci + offsetof(CallInfo, top)]);

        // But if it is vararg, update it to 'argi'
        Label skipVararg;

        build.test(byte[proto + offsetof(Proto, is_vararg)], 1);
        build.jcc(ConditionX64::Zero, skipVararg);
        build.mov(rax, argi);

        build.setLabel(skipVararg);

        build.mov(qword[rState + offsetof(lua_State, top)], rax);

        // Switch current code
        // ci->savedpc = p->code;
        build.mov(rax, qword[proto + offsetof(Proto, code)]);
        build.mov(sCode, rax); // note: this needs to be before the next store for optimal performance
        build.mov(qword[ci + offsetof(CallInfo, savedpc)], rax);

        // Switch current constants
        build.mov(rConstants, qword[proto + offsetof(Proto, k)]);

        // Get native function entry
        build.mov(rax, qword[proto + offsetof(Proto, exectarget)]);
        build.test(rax, rax);
        build.jcc(ConditionX64::Zero, helpers.exitContinueVm);

        // Mark call frame as native
        build.mov(dword[ci + offsetof(CallInfo, flags)], LUA_CALLINFO_NATIVE);

        build.jmp(rax);
    }

    build.setLabel(cFuncCall);

    {
        // results = ccl->c.f(L);
        IrCallWrapperX64 cFuncCallWrap(regs, build, instIdx);
        cFuncCallWrap.addArgument(SizeX64::qword, rState);
        cFuncCallWrap.call(qword[ccl + offsetof(Closure, c.f)]);
        RegisterX64 results = eax;

        build.test(results, results);
        build.jcc(ConditionX64::Less, helpers.exitNoContinueVm);

        if (nresults != 0 && nresults != 1)
        {
            IrCallWrapperX64 callEpilogCWrap(regs, build, instIdx);
            callEpilogCWrap.addArgument(SizeX64::qword, rState);
            callEpilogCWrap.addArgument(SizeX64::dword, OperandX64(nresults));
            callEpilogCWrap.addArgument(SizeX64::dword, results);
            callEpilogCWrap.call(qword[rNativeContext + offsetof(NativeContext, callEpilogC)]);

            emitUpdateBase(build);
            return;
        }

        RegisterX64 ci = rdx;
        RegisterX64 cip = rcx;
        RegisterX64 vali = rsi;

        build.mov(ci, qword[rState + offsetof(lua_State, ci)]);
        build.lea(cip, addr[ci - sizeof(CallInfo)]);

        // L->base = cip->base
        build.mov(rBase, qword[cip + offsetof(CallInfo, base)]);
        build.mov(qword[rState + offsetof(lua_State, base)], rBase);

        if (nresults == 1)
        {
            // Opportunistically copy the result we expected from (L->top - results)
            build.mov(vali, qword[rState + offsetof(lua_State, top)]);
            build.shl(results, kTValueSizeLog2);
            build.sub(vali, qwordReg(results));
            build.vmovups(xmm0, xmmword[vali]);
            build.vmovups(luauReg(ra), xmm0);

            Label skipnil;

            // If there was no result, override the value with 'nil'
            build.test(results, results);
            build.jcc(ConditionX64::NotZero, skipnil);
            build.mov(luauRegTag(ra), LUA_TNIL);
            build.setLabel(skipnil);
        }

        // L->ci = cip
        build.mov(qword[rState + offsetof(lua_State, ci)], cip);

        // L->top = cip->top
        build.mov(rax, qword[cip + offsetof(CallInfo, top)]);
        build.mov(qword[rState + offsetof(lua_State, top)], rax);
    }
}

void emitInstReturn(AssemblyBuilderX64& build, ModuleHelpers& helpers, int ra, int actualResults, bool functionVariadic)
{
    RegisterX64 res = rdi;
    RegisterX64 written = ecx;

    if (functionVariadic)
    {
        build.mov(res, qword[rState + offsetof(lua_State, ci)]);
        build.mov(res, qword[res + offsetof(CallInfo, func)]);
    }
    else if (actualResults != 1)
        build.lea(res, addr[rBase - sizeof(TValue)]); // invariant: ci->func + 1 == ci->base for non-variadic frames

    if (actualResults == 0)
    {
        build.xor_(written, written);
        build.jmp(helpers.return_);
    }
    else if (actualResults == 1 && !functionVariadic)
    {
        // fast path: minimizes res adjustments
        // note that we skipped res computation for this specific case above
        build.vmovups(xmm0, luauReg(ra));
        build.vmovups(xmmword[rBase - sizeof(TValue)], xmm0);
        build.mov(res, rBase);
        build.mov(written, 1);
        build.jmp(helpers.return_);
    }
    else if (actualResults >= 1 && actualResults <= 3)
    {
        for (int r = 0; r < actualResults; ++r)
        {
            build.vmovups(xmm0, luauReg(ra + r));
            build.vmovups(xmmword[res + r * sizeof(TValue)], xmm0);
        }
        build.add(res, actualResults * sizeof(TValue));
        build.mov(written, actualResults);
        build.jmp(helpers.return_);
    }
    else
    {
        RegisterX64 vali = rax;
        RegisterX64 valend = rdx;

        // vali = ra
        build.lea(vali, luauRegAddress(ra));

        // Copy as much as possible for MULTRET calls, and only as much as needed otherwise
        if (actualResults == LUA_MULTRET)
            build.mov(valend, qword[rState + offsetof(lua_State, top)]); // valend = L->top
        else
            build.lea(valend, luauRegAddress(ra + actualResults)); // valend = ra + actualResults

        build.xor_(written, written);

        Label repeatValueLoop, exitValueLoop;

        if (actualResults == LUA_MULTRET)
        {
            build.cmp(vali, valend);
            build.jcc(ConditionX64::NotBelow, exitValueLoop);
        }

        build.setLabel(repeatValueLoop);
        build.vmovups(xmm0, xmmword[vali]);
        build.vmovups(xmmword[res], xmm0);
        build.add(vali, sizeof(TValue));
        build.add(res, sizeof(TValue));
        build.inc(written);
        build.cmp(vali, valend);
        build.jcc(ConditionX64::Below, repeatValueLoop);

        build.setLabel(exitValueLoop);
        build.jmp(helpers.return_);
    }
}

void emitInstSetList(IrRegAllocX64& regs, AssemblyBuilderX64& build, int ra, int rb, int count, uint32_t index, int knownSize, uint32_t instIdx)
{

    OperandX64 last = index + count - 1;

    // Using non-volatile 'rbx' for dynamic 'count' value (for LUA_MULTRET) to skip later recomputation
    // We also keep 'count' scaled by sizeof(TValue) here as it helps in the loop below
    RegisterX64 cscaled = rbx;

    if (count == LUA_MULTRET)
    {
        RegisterX64 tmp = rax;

        // count = L->top - rb
        build.mov(cscaled, qword[rState + offsetof(lua_State, top)]);
        build.lea(tmp, luauRegAddress(rb));
        build.sub(cscaled, tmp); // Using byte difference

        // L->top = L->ci->top
        build.mov(tmp, qword[rState + offsetof(lua_State, ci)]);
        build.mov(tmp, qword[tmp + offsetof(CallInfo, top)]);
        build.mov(qword[rState + offsetof(lua_State, top)], tmp);

        // last = index + count - 1;
        last = edx;
        build.mov(last, dwordReg(cscaled));
        build.shr(last, kTValueSizeLog2);
        build.add(last, index - 1);
    }

    RegisterX64 table = regs.takeReg(rax, kInvalidInstIdx);

    build.mov(table, luauRegValue(ra));

    if (count == LUA_MULTRET || knownSize < 0 || knownSize < int(index + count - 1))
    {
        Label skipResize;

        // Resize if h->sizearray < last
        build.cmp(dword[table + offsetof(LuaTable, sizearray)], last);
        build.jcc(ConditionX64::NotBelow, skipResize);

        IrCallWrapperX64 resizeCallWrap(regs, build, instIdx);
        resizeCallWrap.addArgument(SizeX64::qword, rState);
        resizeCallWrap.addArgument(SizeX64::qword, table);
        resizeCallWrap.addArgument(SizeX64::dword, last); 
        resizeCallWrap.call(qword[rNativeContext + offsetof(NativeContext, luaH_resizearray)]);
        build.mov(table, luauRegValue(ra));


        build.setLabel(skipResize);
    }

    RegisterX64 arrayDst = rdx;
    RegisterX64 offset = rcx;

    build.mov(arrayDst, qword[table + offsetof(LuaTable, array)]);

    const int kUnrollSetListLimit = 4;

    if (count != LUA_MULTRET && count <= kUnrollSetListLimit)
    {
        for (int i = 0; i < count; ++i)
        {
            // setobj2t(L, &array[index + i - 1], rb + i);
            build.vmovups(xmm0, luauRegValue(rb + i));
            build.vmovups(xmmword[arrayDst + (index + i - 1) * sizeof(TValue)], xmm0);
        }
    }
    else
    {
        CODEGEN_ASSERT(count != 0);

        build.xor_(offset, offset);
        if (index != 1)
            build.add(arrayDst, (index - 1) * sizeof(TValue));

        Label repeatLoop, endLoop;
        OperandX64 limit = count == LUA_MULTRET ? cscaled : OperandX64(count * sizeof(TValue));

        // If c is static, we will always do at least one iteration
        if (count == LUA_MULTRET)
        {
            build.cmp(offset, limit);
            build.jcc(ConditionX64::NotBelow, endLoop);
        }

        build.setLabel(repeatLoop);

        // setobj2t(L, &array[index + i - 1], rb + i);
        build.vmovups(xmm0, xmmword[offset + rBase + rb * sizeof(TValue)]); // luauReg(rb) unwrapped to add offset
        build.vmovups(xmmword[offset + arrayDst], xmm0);

        build.add(offset, sizeof(TValue));
        build.cmp(offset, limit);
        build.jcc(ConditionX64::Below, repeatLoop);

        build.setLabel(endLoop);
    }

    callBarrierTableFast(regs, build, table, {});
}

void emitInstForGLoop(IrRegAllocX64& regs, AssemblyBuilderX64& build, int ra, int aux, Label& loopRepeat, uint32_t instIdx)
{
    // ipairs-style traversal is handled in IR
    CODEGEN_ASSERT(aux >= 0);



    RegisterX64 tableValReg = regs.takeReg(rcx, instIdx);
    RegisterX64 indexValReg = regs.takeReg(rdx, instIdx);

    build.mov(tableValReg, luauRegValue(ra + 1));
    build.mov(indexValReg, luauRegValue(ra + 2));

    RegisterX64 elemPtr = rax;

    // &array[index]
    build.mov(dwordReg(elemPtr), dwordReg(indexValReg));
    build.shl(dwordReg(elemPtr), kTValueSizeLog2);
    build.add(elemPtr, qword[tableValReg + offsetof(LuaTable, array)]);

    // Clear extra variables since we might have more than two
    for (int i = 2; i < aux; ++i)
        build.mov(luauRegTag(ra + 3 + i), LUA_TNIL);

    Label skipArray, skipArrayNil;

    // First we advance index through the array portion
    // while (unsigned(index) < unsigned(sizearray))
    Label arrayLoop = build.setLabel();
    build.cmp(dwordReg(indexValReg), dword[tableValReg + offsetof(LuaTable, sizearray)]);
    build.jcc(ConditionX64::NotBelow, skipArray);

    // If element is nil, we increment the index; if it's not, we still need 'index + 1' inside
    build.inc(indexValReg);

    build.cmp(dword[elemPtr + offsetof(TValue, tt)], LUA_TNIL);
    build.jcc(ConditionX64::Equal, skipArrayNil);

    // setpvalue(ra + 2, reinterpret_cast<void*>(uintptr_t(index + 1)), LU_TAG_ITERATOR);
    build.mov(luauRegValue(ra + 2), indexValReg);
    // Extra should already be set to LU_TAG_ITERATOR
    // Tag should already be set to lightuserdata

    // setnvalue(ra + 3, double(index + 1));
    build.vcvtsi2sd(xmm0, xmm0, dwordReg(indexValReg)); 
    build.vmovsd(luauRegValue(ra + 3), xmm0);
    build.mov(luauRegTag(ra + 3), LUA_TNUMBER);

    // setobj2s(L, ra + 4, e);
    setLuauReg(build, xmm2, ra + 4, xmmword[elemPtr]);

    build.jmp(loopRepeat);

    build.setLabel(skipArrayNil);

    // Index already incremented, advance to next array element
    build.add(elemPtr, sizeof(TValue));
    build.jmp(arrayLoop);

    build.setLabel(skipArray);

    // Call helper to assign next node value or to signal loop exit
    IrCallWrapperX64 forgLoopCallWrap(regs, build, instIdx);
    forgLoopCallWrap.addArgument(SizeX64::qword, rState);
    forgLoopCallWrap.addArgument(SizeX64::qword, tableValReg);
    forgLoopCallWrap.addArgument(SizeX64::qword, indexValReg);
    forgLoopCallWrap.addArgument(SizeX64::qword, luauRegAddress(ra));
    forgLoopCallWrap.call(qword[rNativeContext + offsetof(NativeContext, forgLoopNodeIter)]);

    regs.freeReg(tableValReg);
    regs.freeReg(indexValReg);

    build.test(al, al);
    build.jcc(ConditionX64::NotZero, loopRepeat);
}

} // namespace X64
} // namespace CodeGen
} // namespace Luau
