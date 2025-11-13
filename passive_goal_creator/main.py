from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from textwrap import dedent
import argparse

from settings import Settings

class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"


class PassiveGoalCreator:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(dedent("""
ユーザーの入力を分析し、明確で実行可能な目標を生成してください。

要件:
1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。
2. あなたが実行可能な行動は以下の行動だけです。
   - インターネットを利用して、目標を達成するための調査を行う。
   - ユーザーのためのレポートを生成する。
   - 構造体の値には日本語で回答を記載する。
   - 内容は具体的で実行可能。
3. 決して2.以外の行動を取ってはいけません。

ユーザーの入力: {query}
"""))

        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})


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
        temperature=0.3
    )
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)

    print(f"{result.text}")


if __name__ == "__main__":

    main()