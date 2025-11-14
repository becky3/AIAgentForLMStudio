import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent, AgentState
from langgraph.graph import END, StateGraph
from passive_goal_creator.main import Goal, PassiveGoalCreator
from prompt_optimizer.main import OptimizedGoal, PromptOptimizer
from pydantic import BaseModel, Field
from response_optimizer.main import ResponseOptimizer


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_length=3,
        max_length=5,
        description="3~5個に分解されたタスク",
    )


class SinglePathPlanGenerationState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        # ChatPromptTemplate を使わずプレーン文字列で生成
        prompt_text = f"""
以下の目標を、AI が実行できるタスクだけに 3〜5 個で分解してください。

【AI ができる範囲】
- TavilySearch による Web 検索
- 情報の抽出・要約・整理

【禁止】
- ゲームを起動・プレイする行為
- スクリーンショット取得など物理行動
- タスクに存在しない固有名詞・ゲーム名を追加すること
- 架空情報を作ること

【出力】
純粋な JSON のみ。余計な文章禁止：

{{
  "values": ["タスク1", "タスク2", "タスク3"]
}}

【目標】  
{query}
"""

        # モデル呼び出し
        raw = self.llm.invoke([HumanMessage(content=prompt_text)]).content

        import json
        import re

        # JSON 部分だけ抽出
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if not json_match:
            raise ValueError(f"JSON が抽出できません:\n{raw}")

        data = json.loads(json_match.group())
        return DecomposedTasks(values=data["values"])



class TaskExecutor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.search = TavilySearch(max_results=5)

    def run(self, task: str) -> str:
        """
        1. LLM に「検索クエリをそのまま抽出」させる
        2. Tavily に直接渡す（ReAct を禁止）
        3. 結果を LLM にまとめさせる
        """

        # ① タスクから検索キーワードを抽出させる（抽象化禁止）
        extract_prompt = f"""
あなたはタスクをそのまま実行するエージェントです。

【ルール】
- 検索クエリはタスク文にある語句だけを使う
- タスクにない固有名詞・作品名・ハード名を追加しない
- 抽象語（比較・最短・信頼性・ステップ数など）を追加しない
- 追加のタスクや推測を行わない
- 架空情報を作らない

【検索クエリ生成】
- タスクに出てきた語句を並べるだけの「短いクエリ」を使う
例：  
  タスク：「ファミコン版ドラクエ1のクリアルート情報を集める」  
  → クエリ：「ファミコン ドラクエ1 クリアルート攻略」

【出力形式】
---
【検索クエリ】  
- 〜  

【要点】  
- 〜  

【タスク結果】  
- 〜  
---

【タスク】  
{task}
"""

        search_query = self.llm.invoke([HumanMessage(content=extract_prompt)]).content.strip()

        # ② Tavily を直接呼ぶ（ReActなし）
        search_result = self.search.invoke({"query": search_query})

        # ③ 結果を LLM で要約（必要な情報のみ）
        summarize_prompt = f"""
次の検索結果を要約し、タスク達成に必要な情報だけを抽出してください。

【タスク】
{task}

【検索クエリ】
{search_query}

【検索結果】
{search_result}

【要件】
- リメイク版とFC版が混ざれば区別して記載
- 不明な情報は推測しない
- 600〜1000文字で簡潔にまとめる
"""

        summary = self.llm.invoke([HumanMessage(content=summarize_prompt)]).content
        return summary

class ResultAggregator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str, response_definition: str, results: list[str]) -> str:
        # 各タスク結果を少し短く切る（LM Studio が落ちないようにするための保険）
        MAX_PER_RESULT_CHARS = 1500
        truncated_results = [r[:MAX_PER_RESULT_CHARS] for r in results]

        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(truncated_results)
        )

        prompt = ChatPromptTemplate.from_template(
"""
            以下の情報をもとに、事実のみをまとめて回答してください。
        推測・補完・創作は禁止します。

        【目標】
        {query}

        【調査結果】
        {results}

        【出力ルール】
        - 調査で得られた事実のみを要約する
        - 言及されていない固有名詞を出さない
        - 架空の情報を作らない

        【最終回答】
"""
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": results_str,
                "response_definition": response_definition,
            }
        )


class SinglePathPlanGeneration:
    def __init__(self, llm: ChatOpenAI):
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.query_decomposer = QueryDecomposer(llm=llm)
        self.task_executor = TaskExecutor(llm=llm)
        self.result_aggregator = ResultAggregator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        graph = StateGraph(SinglePathPlanGenerationState)
        graph.add_node("goal_setting", self._goal_setting)
        graph.add_node("decompose_query", self._decompose_query)
        graph.add_node("execute_task", self._execute_task)
        graph.add_node("aggregate_results", self._aggregate_results)
        graph.set_entry_point("goal_setting")
        graph.add_edge("goal_setting", "decompose_query")
        graph.add_edge("decompose_query", "execute_task")
        graph.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"},
        )
        graph.add_edge("aggregate_results", END)
        return graph.compile()

    def _goal_setting(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        # プロンプト最適化
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        # レスポンス最適化
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)
        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(
            query=state.optimized_goal
        )
        return {"tasks": decomposed_tasks.values}

    def _execute_task(self, state: SinglePathPlanGenerationState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(
        self, state: SinglePathPlanGenerationState
    ) -> dict[str, Any]:
        final_output = self.result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}

    def run(self, query: str) -> str:
        initial_state = SinglePathPlanGenerationState(query=query)
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 1000})
        return final_state.get("final_output", "Failed to generate a final response.")


def main():
    import argparse

    from settings import Settings
    from dotenv import load_dotenv

    load_dotenv()

    settings = Settings()

    parser = argparse.ArgumentParser(
        description="SinglePathPlanGenerationを使用してタスクを実行します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url=settings.LMSTUDIO_URL,
        api_key="lm-studio",
        model=settings.LMSTUDIO_MODEL,
        temperature=0.1
    )
    agent = SinglePathPlanGeneration(llm=llm)
    result = agent.run(args.task)
    print(result)


if __name__ == "__main__":
    main()
