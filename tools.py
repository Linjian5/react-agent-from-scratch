"""ReAct Agent 的工具集。每个工具接收一个字符串输入，返回字符串/可序列化结果。"""
import json


class ToolError(Exception):
    """可恢复的工具错误，会作为 Observation 回喂给 LLM。"""


def calculator(expression: str) -> str:
    """计算算术表达式，仅支持数字和 + - * / ( ) ."""
    allowed = set("0123456789+-*/(). ")
    bad = [c for c in expression if c not in allowed]
    if bad:
        raise ToolError(f"表达式含非法字符 {bad!r}，仅支持数字和 + - * / ( ) .")
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except ZeroDivisionError:
        raise ToolError("除以零")
    except Exception as e:
        raise ToolError(f"计算失败 {expression!r}: {e}")


# 一个小型城市气温知识库（mock），ReAct 演示"查温度 → 算温差"的多步推理
_CITY_TEMP = {
    "北京": 18,
    "上海": 22,
    "广州": 28,
    "深圳": 29,
    "杭州": 23,
    "成都": 20,
    "哈尔滨": 8,
}


def get_temperature(city: str) -> str:
    """查询城市当前气温（摄氏度，mock 数据）。"""
    city = city.strip().strip("\"'")
    if city not in _CITY_TEMP:
        raise ToolError(f"未知城市 '{city}'，已知：{', '.join(_CITY_TEMP)}")
    return f"{city} 当前气温 {_CITY_TEMP[city]}°C"


def wiki_search(query: str) -> str:
    """搜索维基百科式知识（mock，演示信息抽取）。"""
    db = {
        "Apple Remote": "Apple Remote 最初为 Front Row 媒体中心设计。Front Row 已停止维护。",
        "Front Row": "Front Row 是 Apple 的多媒体应用，可用键盘或 Apple Remote 控制。",
        "ReAct": "ReAct 是 Yao et al. 2022 提出的 Agent 范式，结合推理(Reasoning)与行动(Acting)。",
    }
    for k, v in db.items():
        if k.lower() in query.lower():
            return v
    return f"未找到关于 '{query}' 的条目"


# 工具注册表：name -> {fn, description}
TOOL_REGISTRY = {
    "calculator": {
        "fn": calculator,
        "description": "计算算术表达式，输入如 '29 - 18'，仅支持 + - * / ( ) .",
    },
    "get_temperature": {
        "fn": get_temperature,
        "description": "查询城市当前气温，输入城市名如 '北京'",
    },
    "wiki_search": {
        "fn": wiki_search,
        "description": "搜索百科知识，输入查询词如 'Apple Remote'",
    },
}
