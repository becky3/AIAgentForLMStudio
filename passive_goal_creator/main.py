from dotenv import load_dotenv

import argparse

from settings import Settings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from textwrap import dedent
import json, re

class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"


class PassiveGoalCreator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> Goal:

        # ここで prompt_text を安全に生成
        prompt_text = dedent(f"""
以下のユーザー入力を読み取り、ゲーム攻略エージェントとして実行すべき
「シンプルで誤解のない目標(description)」を JSON 形式で生成してください。

【必須ルール】
- ゲームタイトル・ハード名は絶対に変更・翻訳しない
- 攻略内容そのものを生成してはならない（手順・ルート・ボス名などを記述しない）
- 目標は「LLM と検索ツールで実行可能な範囲の作業内容」を定義することだけに限定する
- 構造化された説明・段階・項目リスト・具体的な手順を生成してはならない
- 作品内容に踏み込む表現（イベント名・アイテム名・敵名など）は禁止
- 事実の創作は禁止
- 出力は日本語

【生成すべき description の条件】
- インターネット検索によって情報収集を行い
- ユーザーが指定したゲームタイトルの「クリアルート」を
- 事実に基づいて整理することを目的とした
- 一文の目標だけを記述する

【出力形式（厳守）】
次の JSON のみを返す。説明文は禁止。

{{
  "description": "ここに目標を書く"
}}

ユーザーの入力:
{query}
        """)

        # LLM 呼び出し
        result = self.llm.invoke([HumanMessage(content=prompt_text)]).content

        # JSON だけを抽出
        json_match = re.search(r"\{[\s\S]*\}", result)
        if not json_match:
            raise ValueError(f"JSON抽出失敗:\n{result}")

        data = json.loads(json_match.group())
        return Goal(description=data["description"])


def main():

    load_dotenv()  # .env を自動読み込み

    # 確認用（削除してOK）
    import os
    print("LangChain project:", os.getenv("LANGCHAIN_PROJECT"))

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="PassiveGoalCreatorを利用して目標を生成します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url=settings.LMSTUDIO_URL,
        api_key="lm-studio",
        model=settings.LMSTUDIO_MODEL,
        temperature=0.1
    )
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)

    print(f"{result.text}")


if __name__ == "__main__":

    main()