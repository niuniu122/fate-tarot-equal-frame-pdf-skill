# Fate Tarot Equal-Frame PDF Skill

用于将带装饰外框的塔罗牌、神谕卡或游戏卡原图转换为印刷刀模 PDF 的 Codex Skill。

它会逐张识别最外层主体边框，保留边框及内部完整设计，并仅使用同一张原图中主体外侧的颜色或渐变生成出血。默认输出为 `80 × 120 mm` TrimBox 和四边各 `5 mm` 出血。

## 安装

将本仓库克隆到 Codex Skill 目录：

```powershell
git clone https://github.com/niuniu122/fate-tarot-equal-frame-pdf-skill.git "$HOME/.codex/skills/fate-tarot-equal-frame-pdf"
```

需要 Python 3.10 或更高版本。安装运行依赖：

```powershell
python -m pip install -r "$HOME/.codex/skills/fate-tarot-equal-frame-pdf/requirements.txt"
```

仓库已内置经过固定版本验证的 PDF 构建器和检查器，克隆一个仓库即可运行。`pypdfium2` 会提供独立 PDF 渲染检查；也可以通过 `--pdftoppm` 显式使用 Poppler。只有在替换为另一个已审计的构建器时，才需要使用 `--print-skill-root`。

完整使用规则和安全门禁见 [SKILL.md](SKILL.md)，详细识别流程见 [references/workflow.md](references/workflow.md)。

## 测试

```powershell
python -m pip install -r "$HOME/.codex/skills/fate-tarot-equal-frame-pdf/requirements-dev.txt"
python -m pytest -q "$HOME/.codex/skills/fate-tarot-equal-frame-pdf/tests"
```

默认测试不需要真实卡牌原图。若要运行真实批次回归测试，可设置：

- `FATE_TAROT_SOURCE_DIR`：原始 PNG 所在目录。
- `FATE_TAROT_BATCH_CARDS_DIR`：逐卡 manifest 与验证制品目录。
- `FATE_TAROT_APPROVED_MANIFESTS`：经人工批准的 manifest 路径列表，使用当前系统的路径分隔符连接；仅这些清单会进入全批次导出回归测试。

未设置这些变量时，真实卡牌回归会跳过；一旦显式设置，目录、卡牌或 manifest 缺失会直接使测试失败，避免错误的绿色结果。
