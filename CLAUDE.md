## 项目定位
小而美的 LLM 自主 Agent 助手，不是通用 Agent 框架。

## 项目环境
- 开发环境：uv 包管理（强制）、python：3.12。
- .env管理环境变量。
- github代码管理。

## 开发规范
- 开发模式：TDD 测试驱动开发（强制），任何功能开发，必须先写失败的测试（Red），再写最小代码使测试通过（Green），最后重构代码（Refactor）。
- 高内聚，低耦合：一个类/模块只关注一个职责，通过组合而非内部引用来协作，避免循环依赖和职责混杂。
- 轻量初始化：类构造函数只接收核心依赖，避免层层传递参数、config 值、类实例；非必需依赖按需取用（如通过 Config 单例、延迟加载），降低实例化复杂度。
- 合理的架构设计，轻量、专注、易于扩展。
- **Config 统一配置原则**：`class Config` 是唯一可以设置配置项默认值的地方，其他模块不得定义与 `Config` 字段含义相同的默认值。配置项从 `Config` 读取，而非自行定义降级值。
- **注释规范**：类和方法必须添加 docstring 说明用途。避免对显而易见的代码添加逐行注释，注释应聚焦于设计意图、边界条件、非显而易见的权衡。

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

# 使用中文推理和回复