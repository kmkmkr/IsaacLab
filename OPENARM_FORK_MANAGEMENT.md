# OpenArm フォーク運用ガイド (Isaac Lab v2.3.2 固定)

このリポジトリは [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) のフォークです。
OpenArm 関連の開発を **Isaac Lab v2.3.2 に固定** して進めます。コードのバージョンと実行コンテナの
バージョンを一致させ、再現性を確保することが目的です。

## 固定対象 (Pinned baseline)

| 項目 | 値 |
|---|---|
| Isaac Lab バージョン | v2.3.2 |
| ベースタグ | `v2.3.2` |
| ベースコミット | `37ddf62687` (Bumps version to v2.3.2, #4399) |
| 実行コンテナ | `docker/apptainer/build/isaac-lab-v2.3.2.sif` |

> 上流 `upstream/main` は既に次バージョンへ進行中です。**固定を維持するため main は追従しません。**

## リモート構成

| remote | URL | 用途 |
|---|---|---|
| `origin` | `git@github.com:kmkmkr/IsaacLab.git` | フォーク。push / PR 先 |
| `upstream` | `https://github.com/isaac-sim/IsaacLab.git` | 公式。更新参照・cherry-pick 元 |

## ブランチモデル

```
tag v2.3.2 (37ddf62687)            ← 上流の固定点（不変）
   │
   └─ openarm/v2.3.2_main           ← 長期統合ブランチ（このプロジェクトの "main"）
        ├─ openarm/pickplace-cube    ← 機能ブランチ
        ├─ openarm/<次の機能>         ← ここから分岐 → PR で戻す
        └─ ...

main                                ← upstream/main のミラー（参照専用、作業しない）
```

- **`openarm/v2.3.2_main`**: 常に「v2.3.2 + マージ済み OpenArm 機能」。安定線。
- **`openarm/<feature>`**: 機能開発。必ず `openarm/v2.3.2_main` から分岐し、PR で戻す。
- **`main`**: 公式 `upstream/main` のミラー。ここでは作業しない。

## 新機能の追加フロー

```bash
git switch openarm/v2.3.2_main
git pull                                 # origin/openarm/v2.3.2_main を取得
git switch -c openarm/<feature>          # 統合ブランチから分岐
# ... 作業・commit ...
git push -u origin openarm/<feature>
# GitHub で  openarm/<feature> -> openarm/v2.3.2_main  の PR を作成 → マージ
```

## 守るべきルール

1. **`upstream/main` を丸ごと merge しない。** 固定が壊れる。必要な上流修正だけ取り込む:
   ```bash
   git fetch upstream
   git switch openarm/v2.3.2_main
   git cherry-pick <commit-sha>
   ```
2. **再現したい時点にはタグを打つ。** 統合ブランチは動くので、デプロイ・検証時点を固定する:
   ```bash
   git tag openarm-v2.3.2-r1
   git push origin openarm-v2.3.2-r1
   ```
3. **フォークのデフォルトブランチを `openarm/v2.3.2_main` にする** (GitHub: Settings → Branches)。
   PR 宛先・clone 初期ブランチが揃い、`main` への誤作業を防げる。
4. **コミット名義を確認する。** 本リポジトリは repo-local 設定で `kmkmkr <xg.techer@gmail.com>`。
   `git log -1 --format='%an <%ae>'` で確認。CI/サーバ上では auto 設定の名義になりやすいので注意。
5. **大きなバイナリ・キャッシュをコミットしない。** `*.sif`、`docker/apptainer/build/tmp/`、
   `nice-dcv-*`、`__pycache__/` などは追跡対象外に保つ。

## Isaac Lab のバージョンを上げる時

v2.3.2 を捨てて新バージョンへ移る時は、**既存ブランチを書き換えず新規に作る**:

```bash
git fetch upstream --tags
git switch -c openarm/v2.4.0_main v2.4.0     # 新バージョンのタグから
# OpenArm の各機能を cherry-pick / 再適用して移植
```

`openarm/v2.3.2_main` はそのまま残るので、いつでも旧固定環境に戻れる。
ブランチ名に基準バージョンを含めるのは、この「バージョン移行を明示的な分岐にする」ための設計。
