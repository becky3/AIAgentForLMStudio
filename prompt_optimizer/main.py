from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from passive_goal_creator.main import Goal, PassiveGoalCreator
from pydantic import BaseModel, Field

class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準: {self.metrics})"


class PromptOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            """
            あなたは SMART 原則に基づき、曖昧な目標を
            「1つの具体的行動」と「測定可能な数値指標」に変換する専門家です。

            以下の2つを必ず明確に分離して出力してください：
            1. description：実行可能で非常に具体的な1つの行動目標
            2. metrics：数値基準の列挙（key=value 形式、2個以上）

            =====================================================================
            【description の厳格制約 — GPT-OSS-120B 用】
            =====================================================================
            ■ description は 1 文のみ。  
            ■ 抽象語・曖昧語は絶対禁止。（以下は禁止ワード）
              「まとめる」「分析する」「整理する」「比較する」「改善する」
              「理解する」「把握する」「調査する」「作成する」「最適化する」
              「レポート」「資料」「内容」「情報」「工程」「プロセス」
            ■ 以下の“実行可能動詞ホワイトリスト”以外の動詞は禁止：
              ・取得する  
              ・抽出する（対象が具体的な場合のみ）  
              ・列挙する  
              ・記録する  
              ・分類する  
              ・計測する  
              ・選択する  
            ■ 複数アクションを入れないこと。「単一の中心行動」に限定する。
            ■ 期限は元目標にない限り追加しない。

            =====================================================================
            【metrics の厳格制約 — GPT-OSS-120B 用】
            =====================================================================
            ■ key=value 形式の数値基準を 2 つ以上含めること。
            ■ 文章禁止。数値指標のみを列挙すること。
            ■ 使用可能キー（必要なら使うこと）：
              site_count, item_count, step_count, page_count, char_count, word_count

            =====================================================================
            【JSON 出力制約（重要）】
            =====================================================================
            ■ 出力は **純粋な JSON のみ**。
            ■ 前後にテキスト・説明・コードブロックは禁止。
            ■ これに違反した場合、評価は 0（zero）である。

            出力フォーマット：
            {{
                "description": "string",
                "metrics": "string"
            }}

            =====================================================================
            【元の目標】
            {query}

            上記すべてに従い、純粋な JSON のみで返答してください。
            """
        )
        response = (prompt | self.llm).invoke({"query": query})
        text = response.content

        import json
        data = json.loads(text)

        return OptimizedGoal(**data)


def main():
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    from settings import Settings

    settings = Settings()
    parser = argparse.ArgumentParser(
        description="PromptOptimizerを利用して、生成された目標のリストを最適化します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url=settings.LMSTUDIO_URL,
        api_key="lm-studio",
        model=settings.LMSTUDIO_MODEL,
        temperature=0.3
    )

    passive_goal_creator = PassiveGoalCreator(llm=llm)
    goal: Goal = passive_goal_creator.run(query=args.task)

    prompt_optimizer = PromptOptimizer(llm=llm)
    optimised_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)

    print(f"{optimised_goal.text}")


if __name__ == "__main__":
    main()