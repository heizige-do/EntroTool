"""Generate a self-contained HTML audit view of saved tool merges."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(value)
    return rows


def load_source_tools(path: Path) -> dict[str, dict[str, Any]]:
    raw_tools = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_tools, list):
        raise ValueError(f"Expected a list of tools in {path}")
    tools: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_tools):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or raw.get("path") or "").strip()
        if not name:
            continue
        uid = f"{name}@@{index:05d}"
        tools[uid] = {**raw, "uid": uid}
    return tools


def resolve_tools_file(dataset: str, override: str | None) -> Path:
    if override:
        return Path(override)
    config = json.loads((PROJECT / "config.json").read_text(encoding="utf-8"))
    dataset_config = config.get("evaluation", {}).get("datasets", {}).get(dataset)
    if not isinstance(dataset_config, Mapping) or not dataset_config.get("tools"):
        raise KeyError(f"Unknown dataset {dataset!r}; pass --tools explicitly")
    return WORKSPACE / "datasets" / dataset / str(dataset_config["tools"])


def arguments_of(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = tool.get("arguments", {})
    return arguments if isinstance(arguments, Mapping) else {}


def signature_of(tool: Mapping[str, Any]) -> str:
    saved = str(tool.get("signature", "")).strip()
    if saved:
        return saved
    name = str(tool.get("name") or tool.get("path") or "unnamed_tool")
    arguments = arguments_of(tool)
    properties = arguments.get("properties", {})
    required = set(arguments.get("required", []))
    if not isinstance(properties, Mapping):
        properties = {}
    parts: list[str] = []
    for parameter, raw_spec in properties.items():
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        optional = "" if parameter in required else " = None"
        parts.append(f"{parameter}: {spec.get('type', 'any')}{optional}")
    return f"{name}({', '.join(parts)})"


def description_of(tool: Mapping[str, Any]) -> str:
    return " ".join(
        str(tool.get(key, "")).strip()
        for key in ("description", "usage", "functionality")
        if str(tool.get(key, "")).strip()
    )


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def compact_description(tool: Mapping[str, Any], limit: int = 150) -> str:
    description = " ".join(description_of(tool).split())
    if len(description) <= limit:
        return description
    return description[: limit - 1].rstrip() + "…"


def tool_node(tool: Mapping[str, Any], tone: str) -> str:
    description = compact_description(tool)
    name = escape(tool.get("name") or tool.get("path") or "unnamed_tool")
    description_html = f"<p>{escape(description)}</p>" if description else ""
    return (
        f'<article class="tool-node {tone}">'
        f"<strong>{name}</strong>"
        f"<code>{escape(signature_of(tool))}</code>"
        f"{description_html}"
        "</article>"
    )


def build_groups(
    rows: list[dict[str, Any]], source_tools: Mapping[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    groups: list[dict[str, Any]] = []
    missing_count = 0
    for row in rows:
        corrected = row.get("corrected_groups", [])
        canonical = row.get("canonical_tools", [])
        if not isinstance(corrected, list) or not isinstance(canonical, list):
            continue
        if len(corrected) != len(canonical):
            raise ValueError(
                f"{row.get('component_id')}: corrected_groups and canonical_tools differ in length"
            )
        initial_members = [str(uid) for uid in row.get("members", [])]
        for group_index, (member_uids, merged_tool) in enumerate(
            zip(corrected, canonical)
        ):
            uids = [str(uid) for uid in member_uids]
            sources = []
            for uid in uids:
                source = source_tools.get(uid)
                if source is None:
                    missing_count += 1
                    source = {
                        "uid": uid,
                        "name": uid,
                        "description": "Source tool not found",
                    }
                sources.append(source)
            groups.append(
                {
                    "component_id": str(row.get("component_id", "")),
                    "group_index": group_index,
                    "initial_size": len(initial_members),
                    "autocorrection_enabled": bool(
                        row.get("autocorrection_enabled", False)
                    ),
                    "member_uids": uids,
                    "sources": sources,
                    "merged_tool": merged_tool if isinstance(merged_tool, dict) else {},
                }
            )
    return groups, missing_count


def render_html(
    dataset: str,
    merge_dir: Path,
    groups: list[dict[str, Any]],
    input_tools: int,
    merged_tools: int,
    missing_count: int,
) -> str:
    merged_groups = sum(len(group["sources"]) > 1 for group in groups)
    component_count = len({group["component_id"] for group in groups})
    sections: list[str] = []
    for group in groups:
        sources = group["sources"]
        merged_tool = group["merged_tool"]
        searchable = " ".join(
            [group["component_id"], str(merged_tool.get("name", ""))]
            + [str(tool.get("name", "")) for tool in sources]
        ).lower()
        source_html = "".join(tool_node(tool, "source") for tool in sources)
        component = escape(group["component_id"])
        group_number = group["group_index"] + 1
        target_html = tool_node(merged_tool, "merged")
        sections.append(
            f'<section class="merge-group" data-size="{len(sources)}" '
            f'data-search="{escape(searchable)}">'
            '<header class="group-header">'
            f"<span>{component} · group {group_number}</span>"
            f"<b>{len(sources)} → 1</b>"
            "</header>"
            '<div class="merge-flow">'
            f'<div class="source-lane">{source_html}</div>'
            '<div class="flow-arrow" aria-hidden="true"><span>→</span></div>'
            f'<div class="target-lane">{target_html}</div>'
            "</div></section>"
        )
    reduction = input_tools - merged_tools
    reduction_rate = (reduction / input_tools * 100.0) if input_tools else 0.0
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(dataset)} merge map</title>
  <style>
    :root {{ --ink:#17211d; --muted:#66736d; --line:#cbd5cf; --white:#fff; --source:#2266a2; --merged:#138262; --accent:#d99b16; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:#edf2ee; font-family:"Segoe UI",sans-serif; }}
    .shell {{ width:min(1600px,calc(100% - 28px)); margin:0 auto; padding:22px 0 50px; }}
    h1,h2,h3,h4,p {{ margin-top:0; }}
    h1 {{ font:700 30px Georgia,serif; margin-bottom:4px; }}
    .subtitle {{ color:var(--muted); margin-bottom:16px; font-size:12px; overflow-wrap:anywhere; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:1px; border:1px solid var(--line); background:var(--line); }}
    .metric {{ flex:1 1 130px; padding:10px 14px; background:var(--white); }}
    .metric strong {{ display:inline; margin-right:6px; font:700 19px Georgia,serif; }} .metric span {{ color:var(--muted); font-size:11px; }}
    .controls {{ position:sticky; top:0; z-index:10; display:flex; gap:12px; align-items:center; padding:9px 0; background:rgba(237,242,238,.96); backdrop-filter:blur(8px); }}
    input[type=search] {{ flex:1; min-width:0; border:1px solid #99a79f; background:white; padding:8px 10px; font:inherit; font-size:13px; }}
    label {{ display:flex; gap:8px; align-items:center; white-space:nowrap; }}
    #visible-count {{ color:var(--muted); min-width:95px; text-align:right; font-size:12px; }}
    #groups {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }}
    .merge-group {{ border:1px solid var(--line); background:var(--white); box-shadow:0 3px 10px rgba(25,45,35,.04); min-width:0; }}
    .group-header {{ display:flex; justify-content:space-between; padding:5px 9px; border-bottom:1px solid #e1e7e3; color:var(--muted); font:10px Consolas,monospace; }}
    .group-header b {{ color:var(--merged); }}
    .merge-flow {{ display:grid; grid-template-columns:minmax(0,1fr) 34px minmax(190px,.55fr); align-items:stretch; min-height:76px; }}
    .source-lane {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:4px; padding:5px; align-content:center; background:#f7f9f7; }}
    .target-lane {{ display:flex; align-items:center; padding:5px; background:#f2faf6; }}
    .flow-arrow {{ position:relative; display:flex; align-items:center; justify-content:center; color:var(--accent); background:linear-gradient(90deg,#f7f9f7,#f2faf6); }}
    .flow-arrow::before {{ content:""; position:absolute; left:0; right:0; top:50%; height:2px; background:var(--accent); opacity:.65; }}
    .flow-arrow span {{ position:relative; padding:2px 3px 4px; background:#f4f9f5; font-size:20px; line-height:1; }}
    .tool-node {{ min-width:0; padding:6px 7px; border:1px solid var(--line); border-left:3px solid; background:var(--white); }}
    .tool-node.source {{ border-left-color:var(--source); }} .tool-node.merged {{ width:100%; border-left-color:var(--merged); }}
    .tool-node strong {{ display:block; margin-bottom:2px; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .tool-node code {{ display:block; color:#34423b; font:9px Consolas,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .tool-node p {{ margin:3px 0 0; color:var(--muted); font-size:9px; line-height:1.25; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .warning {{ margin-top:12px; color:#9a3e28; }} .empty {{ padding:50px; text-align:center; color:var(--muted); }}
    @media (max-width:800px) {{ #groups {{ grid-template-columns:1fr; }} }}
    @media (max-width:700px) {{ .merge-flow {{ grid-template-columns:1fr; }} .flow-arrow {{ min-height:28px; }} .flow-arrow::before {{ top:0; bottom:0; left:50%; width:2px; height:auto; }} .flow-arrow span {{ transform:rotate(90deg); }} .target-lane {{ justify-content:center; }} }}
    @media (max-width:560px) {{ .shell {{ width:min(100% - 14px,1600px); padding-top:14px; }} .controls {{ flex-wrap:wrap; }} #visible-count {{ text-align:left; }} .source-lane {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <h1>{escape(dataset)} Tool Merge Map</h1>
    <p class="subtitle">Source tools → canonical tool · {escape(merge_dir)}</p>
    <div class="metrics">
      <div class="metric"><strong>{input_tools}</strong><span>Input tools</span></div>
      <div class="metric"><strong>{merged_tools}</strong><span>Output tools</span></div>
      <div class="metric"><strong>{reduction}</strong><span>Tools removed ({reduction_rate:.1f}%)</span></div>
      <div class="metric"><strong>{component_count}</strong><span>Processed components</span></div>
      <div class="metric"><strong>{merged_groups}</strong><span>Non-singleton merged groups</span></div>
    </div>
    {f'<p class="warning">Warning: {missing_count} source tool references could not be resolved.</p>' if missing_count else ''}
    <div class="controls">
      <input id="search" type="search" placeholder="Search component or tool name" aria-label="Search merges">
      <label><input id="merged-only" type="checkbox" checked> Only actual merges</label>
      <span id="visible-count"></span>
    </div>
    <div id="groups">{''.join(sections) if sections else '<div class="empty">No merge groups found.</div>'}</div>
  </main>
  <script>
    const groups = [...document.querySelectorAll('.merge-group')];
    const search = document.getElementById('search');
    const mergedOnly = document.getElementById('merged-only');
    const visibleCount = document.getElementById('visible-count');
    function filterGroups() {{
      const query = search.value.trim().toLowerCase();
      let visible = 0;
      for (const group of groups) {{
        const matchesText = !query || group.dataset.search.includes(query);
        const matchesSize = !mergedOnly.checked || Number(group.dataset.size) > 1;
        const show = matchesText && matchesSize;
        group.hidden = !show;
        if (show) visible += 1;
      }}
      visibleCount.textContent = `${{visible}} / ${{groups.length}} groups`;
    }}
    search.addEventListener('input', filterGroups);
    mergedOnly.addEventListener('change', filterGroups);
    filterGroups();
  </script>
</body>
</html>
"""


def render_graph_html(
    dataset: str,
    merge_dir: Path,
    groups: list[dict[str, Any]],
    input_tools: int,
    merged_tools: int,
    missing_count: int,
) -> str:
    graph_groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        sources = []
        for tool in group["sources"]:
            sources.append(
                {
                    "name": str(tool.get("name") or tool.get("path") or "unnamed_tool"),
                    "signature": signature_of(tool),
                    "description": compact_description(tool, 220),
                }
            )
        merged_tool = group["merged_tool"]
        graph_groups.append(
            {
                "id": f"g{group_index}",
                "component": group["component_id"],
                "groupIndex": group["group_index"] + 1,
                "sources": sources,
                "target": {
                    "name": str(
                        merged_tool.get("name")
                        or merged_tool.get("path")
                        or "unnamed_tool"
                    ),
                    "signature": signature_of(merged_tool),
                    "description": compact_description(merged_tool, 220),
                },
            }
        )
    graph_json = json.dumps(graph_groups, ensure_ascii=False).replace("</", "<\\/")
    reduction = input_tools - merged_tools
    reduction_rate = (reduction / input_tools * 100.0) if input_tools else 0.0
    template = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__DATASET__ merge graph</title>
    <style>
        :root { --ink:#172033; --muted:#687386; --line:#d9dfe8; --panel:#ffffff; --canvas:#ffffff; }
        * { box-sizing:border-box; }
        html,body { margin:0; height:100%; overflow:hidden; color:var(--ink); font-family:"Segoe UI",sans-serif; }
        body { display:grid; grid-template-rows:auto 1fr; background:#fff; }
        header { z-index:5; padding:10px 16px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); box-shadow:0 2px 14px rgba(39,55,86,.07); transition:padding .2s ease; }
        header.collapsed { padding:7px 12px; }
        header.collapsed .stats,header.collapsed .controls,header.collapsed .legend,header.collapsed h1 { display:none; }
        header.collapsed .top { justify-content:flex-end; }
        .top { display:flex; align-items:center; gap:14px; }
        h1 { margin:0; color:#172033; font:700 21px Georgia,serif; white-space:nowrap; }
        .stats { display:flex; gap:12px; color:var(--muted); font-size:11px; white-space:nowrap; }
        .stats b { color:var(--ink); font-size:13px; }
        .controls { display:flex; flex:1; justify-content:flex-end; align-items:center; gap:8px; }
        input[type=search] { width:min(340px,35vw); padding:7px 9px; border:1px solid #9caaa2; font:inherit; font-size:12px; }
        button { border:1px solid #c5cedb; border-radius:5px; background:white; color:#33415c; padding:7px 10px; cursor:pointer; }
        button:hover { background:#f4f7fb; border-color:#8fa0b8; }
        #toggle-toolbar { margin-left:auto; }
        .radius-control { display:flex; align-items:center; gap:5px; padding-left:7px; border-left:1px solid var(--line); }
        .radius-control.disabled { opacity:.42; pointer-events:none; }
        .radius-control input { width:92px; accent-color:#526b91; }
        .radius-control output { min-width:34px; color:var(--muted); font:11px Consolas,monospace; }
        .radius-control button { width:29px; padding:6px 0; }
        label { display:flex; align-items:center; gap:5px; font-size:12px; white-space:nowrap; }
        .legend { display:flex; gap:14px; margin-top:7px; color:var(--muted); font-size:11px; }
        .legend span { display:flex; align-items:center; gap:5px; }
        .dot { width:11px; height:11px; border-radius:50%; border:2px solid #fff; background:#426b91; box-shadow:0 0 0 1px #426b91; }
        .dot.target { width:14px; height:14px; border:3px double #fff; background:#426b91; box-shadow:0 0 0 1px #426b91; }
        #viewport { position:relative; min-height:0; background-color:var(--canvas); background-image:radial-gradient(#dbe1e9 0.55px,transparent 0.55px); background-size:22px 22px; }
        svg { width:100%; height:100%; display:block; cursor:grab; user-select:none; }
        svg.panning { cursor:grabbing; }
        .cluster-halo { fill:transparent; stroke:none; pointer-events:all; cursor:move; }
        .edge { fill:none; stroke-width:1.45; opacity:.56; }
        .node { cursor:move; }
        .node circle { stroke-width:2; filter:drop-shadow(0 4px 5px rgba(35,53,82,.13)); }
        .node.selected circle { stroke:#f5c451; stroke-width:5; filter:drop-shadow(0 0 7px rgba(245,196,81,.75)); }
        .node.source circle { stroke:rgba(255,255,255,.82); }
        .node.target circle { stroke:#fff; stroke-width:4; }
        .node text { pointer-events:none; }
        .node .name { fill:#fff; font:700 13px "Segoe UI",sans-serif; text-anchor:middle; }
        .node .params { fill:rgba(255,255,255,.92); font:9.5px Consolas,monospace; text-anchor:middle; }
        .node .description { fill:rgba(255,255,255,.78); font:9.5px "Segoe UI",sans-serif; text-anchor:middle; }
        .node.target .name { font-size:14px; }
        .dim { opacity:.08; }
        .matched { opacity:1; }
        #tooltip { position:absolute; display:none; z-index:10; width:min(390px,calc(100vw - 24px)); padding:11px 12px; border:1px solid var(--line); border-left:4px solid var(--tip-color,#2876b5); background:rgba(255,255,255,.98); box-shadow:0 10px 30px rgba(20,40,30,.2); pointer-events:none; }
        #tooltip strong { display:block; margin-bottom:5px; overflow-wrap:anywhere; }
        #tooltip code { display:block; padding:6px; background:#edf2ef; white-space:pre-wrap; overflow-wrap:anywhere; font-size:10px; }
        #tooltip p { margin:7px 0 0; color:var(--muted); font-size:11px; line-height:1.4; }
        #empty { position:absolute; inset:0; display:none; place-items:center; color:var(--muted); }
        .warning { color:#a3442e; }
        @media (max-width:760px) { .top { align-items:flex-start; flex-wrap:wrap; } .controls { width:100%; justify-content:flex-start; } input[type=search] { flex:1; width:auto; } .stats { flex-wrap:wrap; } }
    </style>
</head>
<body>
    <header>
        <div class="top">
            <h1>__DATASET__ Tool Merge Graph</h1>
            <div class="stats">
                <span><b>__INPUT__</b> input</span><span><b>__OUTPUT__</b> output</span><span><b>__REDUCTION__</b> removed (__RATE__%)</span><span><b id="group-count"></b> groups</span>
                __WARNING__
            </div>
            <div class="controls">
                <input id="search" type="search" placeholder="Search tool or component">
                <label><input id="merged-only" type="checkbox" checked> only actual merges</label>
                <button id="fit" title="Fit graph to canvas">Fit</button>
                <button id="zoom-in" title="Zoom in">+</button>
                <button id="zoom-out" title="Zoom out">−</button>
                <div id="radius-control" class="radius-control disabled" title="Select a node to adjust its radius">
                    <button id="radius-down" title="Decrease selected node radius">−</button>
                    <input id="radius" type="range" min="45" max="180" step="1" value="90" aria-label="Selected node radius">
                    <button id="radius-up" title="Increase selected node radius">+</button>
                    <output id="radius-value">--</output>
                </div>
                <button id="export-svg" title="Export current canvas as SVG">SVG</button>
                <button id="export-png" title="Export current canvas as PNG">PNG</button>
            </div>
            <button id="toggle-toolbar" title="Hide toolbar">Hide</button>
        </div>
        <div class="legend"><span><i class="dot"></i>source tool</span><span><i class="dot target"></i>merged canonical tool</span><span>Click a node to resize · Alt + wheel resizes hovered node · nodes cannot overlap</span></div>
    </header>
    <main id="viewport">
        <svg id="graph" aria-label="Tool merge relationship graph">
            <defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="context-stroke"></path></marker></defs>
            <g id="scene"></g>
        </svg>
        <div id="tooltip"></div><div id="empty">No matching merge groups.</div>
    </main>
    <script>
        const allGroups = __GRAPH_DATA__;
        const componentOrder=[...new Set(allGroups.map(group=>group.component))].sort();
        const componentIndex=new Map(componentOrder.map((component,index)=>[component,index]));
        function hslToHex(h,s,l){s/=100;l/=100;const c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs((h/60)%2-1)),m=l-c/2;let r=0,g=0,b=0;if(h<60){r=c;g=x;}else if(h<120){r=x;g=c;}else if(h<180){g=c;b=x;}else if(h<240){g=x;b=c;}else if(h<300){r=x;b=c;}else{r=c;b=x;}return '#'+[r,g,b].map(value=>Math.round((value+m)*255).toString(16).padStart(2,'0')).join('');}
        function componentColor(key){const index=componentIndex.get(key)??0,hue=(index*137.508+17)%360,saturation=68+(index%3)*7,lightness=34+(index%2)*5;return hslToHex(hue,saturation,lightness);}
        const svg = document.getElementById('graph');
        const scene = document.getElementById('scene');
        const viewport = document.getElementById('viewport');
        const tooltip = document.getElementById('tooltip');
        const search = document.getElementById('search');
        const mergedOnly = document.getElementById('merged-only');
        const empty = document.getElementById('empty');
        const radiusControl = document.getElementById('radius-control');
        const radiusInput = document.getElementById('radius');
        const radiusValue = document.getElementById('radius-value');
        let transform = {x:0,y:0,k:1};
        let graphWidth = 1000, graphHeight = 700;
        let nodeElements = [], groupVisuals = [], selectedNode = null;
        const NS = 'http://www.w3.org/2000/svg';
        function el(name, attrs={}) { const node=document.createElementNS(NS,name); for(const [key,value] of Object.entries(attrs)) node.setAttribute(key,value); return node; }
        function short(text, limit) { text=String(text||''); return text.length>limit ? text.slice(0,limit-1)+'…' : text; }
        function parameterText(signature) { const start=signature.indexOf('('), end=signature.lastIndexOf(')'); return start>=0&&end>start ? signature.slice(start+1,end) : ''; }
        function wrapText(text, limit, maxLines=2) { const words=String(text||'').replace(/\\s+/g,' ').trim().split(' '); const lines=[]; let line=''; for(const word of words){ const next=line?line+' '+word:word; if(next.length<=limit){line=next;}else{if(line)lines.push(line);line=word;if(lines.length===maxLines-1)break;} } if(line&&lines.length<maxLines)lines.push(line); const consumed=lines.join(' ').length; if(consumed<String(text||'').trim().length&&lines.length)lines[lines.length-1]=short(lines[lines.length-1],Math.max(4,limit-1))+'…'; return lines; }
        function nodeGeometry(tool,isTarget){const params=parameterText(tool.signature),nameLines=wrapText(tool.name,isTarget?28:23,2),paramLines=wrapText(params||'no parameters',isTarget?34:29,2),descLines=wrapText(tool.description,isTarget?38:32,2),allLines=nameLines.concat(paramLines,descLines),longest=Math.max(...allLines.map(line=>line.length),8),contentLength=String(tool.name||'').length+params.length+String(tool.description||'').length,lineCount=allLines.length,base=46+Math.sqrt(contentLength)*2.2+longest*.72+Math.max(0,lineCount-3)*4,radius=Math.max(58,Math.min(122,base))*(isTarget?1.18:1);return{width:radius*2,height:radius*2,radius,nameLines,paramLines,descLines};}
        function overlaps(a,x,y,b,bx=b.position.x,by=b.position.y,padding=10){return Math.abs(x-bx)<(a.width+b.width)/2+padding&&Math.abs(y-by)<(a.height+b.height)/2+padding;}
        function canPlace(node,x,y,ignored=new Set([node])){return nodeElements.every(other=>ignored.has(other)||!overlaps(node,x,y,other));}
        function canMoveGroup(visual,dx,dy){const moving=new Set(visual.nodes);return visual.nodes.every(node=>nodeElements.every(other=>moving.has(other)||!overlaps(node,node.position.x+dx,node.position.y+dy,other)));}
        function applyTransform() { scene.setAttribute('transform',`translate(${transform.x} ${transform.y}) scale(${transform.k})`); }
        function render() {
            const query=search.value.trim().toLowerCase();
            const groups=allGroups.filter(group => (!mergedOnly.checked || group.sources.length>1) && (!query || JSON.stringify(group).toLowerCase().includes(query)));
            scene.replaceChildren(); nodeElements=[]; groupVisuals=[]; selectedNode=null;syncRadiusControl();empty.style.display=groups.length?'none':'grid'; document.getElementById('group-count').textContent=groups.length;
            if(!groups.length) return;
            const layouts=groups.map(group=>{const sourceGeometries=group.sources.map(tool=>nodeGeometry(tool,false)),targetGeometry=nodeGeometry(group.target,true),maxSourceWidth=Math.max(...sourceGeometries.map(g=>g.width)),maxSourceHeight=Math.max(...sourceGeometries.map(g=>g.height)),count=group.sources.length,radialSpacing=count>1?maxSourceWidth/(2*Math.sin(Math.PI/count))+28:0,targetSpacing=targetGeometry.width/2+maxSourceWidth/2+55,radius=Math.max(190,radialSpacing,targetSpacing);return{sourceGeometries,targetGeometry,radius,width:radius*2+maxSourceWidth+90,height:radius*2+maxSourceHeight+100};});
            const maxNodeWidth=Math.max(...layouts.flatMap(layout=>layout.sourceGeometries.map(g=>g.width).concat(layout.targetGeometry.width))),maxNodeHeight=Math.max(...layouts.flatMap(layout=>layout.sourceGeometries.map(g=>g.height).concat(layout.targetGeometry.height))),maxRadius=Math.max(...layouts.map(layout=>layout.radius)),columns=Math.max(1,Math.ceil(Math.sqrt(groups.length*1.45))),cellW=maxRadius*2+maxNodeWidth*2+180,cellH=maxRadius*2+maxNodeHeight*2+180;
            graphWidth=columns*cellW; graphHeight=Math.ceil(groups.length/columns)*cellH;
            groups.forEach((group,index) => {
                const color=componentColor(group.component), col=index%columns, row=Math.floor(index/columns), cx=col*cellW+cellW/2, cy=row*cellH+cellH/2;
                const sourceCount=group.sources.length, radius=layouts[index].radius, target={x:cx,y:cy};
                const halo=el('rect',{class:'cluster-halo'}); scene.appendChild(halo);
                const positions=group.sources.map((_,i) => { const angle=(Math.PI*2*i/sourceCount)-Math.PI/2; return {x:cx+Math.cos(angle)*radius,y:cy+Math.sin(angle)*radius}; });
                const visual={halo,color,nodes:[],edges:[]}; groupVisuals.push(visual);
                group.sources.forEach((tool,i) => visual.nodes.push(createNode(tool,positions[i],false,color,group,visual)));
                visual.target=createNode(group.target,target,true,color,group,visual); visual.nodes.push(visual.target);
                relaxGroup(visual,target);
                visual.nodes.filter(node=>!node.isTarget).forEach(node => { const line=el('line',{class:'edge',stroke:color,'marker-end':'url(#arrow)'}); scene.insertBefore(line,visual.nodes[0].element); visual.edges.push({line,source:node,target:visual.target}); });
                enableGroupDrag(visual);
                updateGroup(visual);
            });
            fitGraph();
        }
        function createNode(tool,position,isTarget,color,group,visual) {
            const geometry=nodeGeometry(tool,isTarget);
            const g=el('g',{class:`node ${isTarget?'target':'source'}`,transform:`translate(${position.x} ${position.y})`});
            const circle=el('circle',{r:geometry.radius,fill:color});g.appendChild(circle);
            const totalLines=geometry.nameLines.length+geometry.paramLines.length+geometry.descLines.length,gapBeforeParams=5,gapBeforeDescription=6,lineHeight=12,totalHeight=totalLines*lineHeight+gapBeforeParams+gapBeforeDescription,startY=-totalHeight/2+10;
            const addLines=(className,lines,offset)=>{const text=el('text',{x:0,y:startY+offset,class:className});lines.forEach((line,index)=>{const span=el('tspan',{x:0,dy:index?lineHeight:0});span.textContent=line;text.appendChild(span);});g.appendChild(text);return lines.length*lineHeight;};
            let offset=0;offset+=addLines('name',geometry.nameLines,offset);offset+=gapBeforeParams;offset+=addLines('params',geometry.paramLines,offset);offset+=gapBeforeDescription;addLines('description',geometry.descLines,offset);
            g.addEventListener('pointerenter',event => showTooltip(event,tool,isTarget,color)); g.addEventListener('pointermove',moveTooltip); g.addEventListener('pointerleave',hideTooltip);
            const node={element:g,tool,group,position,width:geometry.width,height:geometry.height,baseRadius:geometry.radius,isTarget};
            let dragging=false,last=null;
            g.addEventListener('click',event=>{event.stopPropagation();selectNode(node);});
            g.addEventListener('pointerdown',event=>{event.stopPropagation();selectNode(node);dragging=true;last={x:event.clientX,y:event.clientY};g.setPointerCapture(event.pointerId);hideTooltip();});
            g.addEventListener('pointermove',event=>{if(!dragging)return;const dx=(event.clientX-last.x)/transform.k,dy=(event.clientY-last.y)/transform.k,nextX=node.position.x+dx,nextY=node.position.y+dy;last={x:event.clientX,y:event.clientY};if(!canPlace(node,nextX,nextY))return;node.position.x=nextX;node.position.y=nextY;g.setAttribute('transform',`translate(${node.position.x} ${node.position.y})`);updateGroup(visual);});
            g.addEventListener('pointerup',()=>dragging=false);g.addEventListener('pointercancel',()=>dragging=false);
            g.addEventListener('wheel',event=>{if(!event.altKey)return;event.preventDefault();event.stopPropagation();selectNode(node);resizeNode(node,node.width/2+(event.deltaY<0?4:-4),visual);},{passive:false});
            scene.appendChild(g); nodeElements.push(node); return node;
        }
        function selectNode(node){if(selectedNode)selectedNode.element.classList.remove('selected');selectedNode=node;if(node)node.element.classList.add('selected');syncRadiusControl();}
        function syncRadiusControl(){const enabled=!!selectedNode;radiusControl.classList.toggle('disabled',!enabled);radiusInput.disabled=!enabled;if(enabled){const radius=Math.round(selectedNode.width/2);radiusInput.value=String(radius);radiusValue.value=radius+' px';radiusValue.textContent=radius+' px';}else{radiusValue.value='--';radiusValue.textContent='--';}}
        function resizeNode(node,requestedRadius,visual){const current=node.width/2,target=Math.max(45,Math.min(180,requestedRadius));if(target>current){const probe={...node,width:target*2,height:target*2};if(!nodeElements.every(other=>other===node||!overlaps(probe,node.position.x,node.position.y,other)))return false;}node.width=target*2;node.height=target*2;node.element.querySelector('circle').setAttribute('r',target);const textScale=Math.max(.58,Math.min(1.12,target/node.baseRadius));node.element.querySelectorAll('text').forEach(text=>text.setAttribute('transform',`scale(${textScale})`));updateGroup(visual);syncRadiusControl();return true;}
        function boundaryPoint(from,to,node){const dx=to.x-from.x,dy=to.y-from.y;if(!dx&&!dy)return{x:from.x,y:from.y};const scale=1/Math.max(Math.abs(dx)/(node.width/2+7),Math.abs(dy)/(node.height/2+7));return{x:from.x+dx*scale,y:from.y+dy*scale};}
        function relaxGroup(visual,center){for(let iteration=0;iteration<80;iteration++){let changed=false;for(let i=0;i<visual.nodes.length;i++){const a=visual.nodes[i];if(a.isTarget)continue;for(let j=i+1;j<visual.nodes.length;j++){const b=visual.nodes[j];if(!overlaps(a,a.position.x,a.position.y,b,b.position.x,b.position.y,12))continue;const node=b.isTarget?a:b,dx=node.position.x-center.x,dy=node.position.y-center.y,length=Math.hypot(dx,dy)||1;node.position.x+=dx/length*18;node.position.y+=dy/length*18;node.element.setAttribute('transform',`translate(${node.position.x} ${node.position.y})`);changed=true;}}if(!changed)break;}}
        function updateGroup(visual){visual.edges.forEach(edge=>{const a=boundaryPoint(edge.source.position,edge.target.position,edge.source),b=boundaryPoint(edge.target.position,edge.source.position,edge.target);edge.line.setAttribute('x1',a.x);edge.line.setAttribute('y1',a.y);edge.line.setAttribute('x2',b.x);edge.line.setAttribute('y2',b.y);});const pad=25,minX=Math.min(...visual.nodes.map(n=>n.position.x-n.width/2))-pad,maxX=Math.max(...visual.nodes.map(n=>n.position.x+n.width/2))+pad,minY=Math.min(...visual.nodes.map(n=>n.position.y-n.height/2))-pad,maxY=Math.max(...visual.nodes.map(n=>n.position.y+n.height/2))+pad;visual.halo.setAttribute('x',minX);visual.halo.setAttribute('y',minY);visual.halo.setAttribute('width',maxX-minX);visual.halo.setAttribute('height',maxY-minY);}
        function enableGroupDrag(visual){let dragging=false,last=null;const handle=visual.halo,start=event=>{event.stopPropagation();dragging=true;last={x:event.clientX,y:event.clientY};handle.setPointerCapture(event.pointerId);hideTooltip();},move=event=>{if(!dragging)return;const dx=(event.clientX-last.x)/transform.k,dy=(event.clientY-last.y)/transform.k;last={x:event.clientX,y:event.clientY};if(!canMoveGroup(visual,dx,dy))return;visual.nodes.forEach(node=>{node.position.x+=dx;node.position.y+=dy;node.element.setAttribute('transform',`translate(${node.position.x} ${node.position.y})`);});updateGroup(visual);},stop=()=>dragging=false;handle.addEventListener('pointerdown',start);handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',stop);handle.addEventListener('pointercancel',stop);}
        function showTooltip(event,tool,isTarget,color) { tooltip.style.display='block';tooltip.style.setProperty('--tip-color',color);tooltip.innerHTML=`<strong>${escapeHtml(tool.name)}${isTarget?' · merged':''}</strong><code>${escapeHtml(tool.signature)}</code>${tool.description?`<p>${escapeHtml(tool.description)}</p>`:''}`;moveTooltip(event); }
        function moveTooltip(event) { const rect=viewport.getBoundingClientRect(); let x=event.clientX-rect.left+14,y=event.clientY-rect.top+14; if(x+400>rect.width)x=Math.max(8,x-410); if(y+180>rect.height)y=Math.max(8,y-190);tooltip.style.left=x+'px';tooltip.style.top=y+'px'; }
        function hideTooltip(){tooltip.style.display='none';}
        function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
        function fitGraph(){const rect=viewport.getBoundingClientRect(),pad=34;transform.k=Math.min((rect.width-pad*2)/graphWidth,(rect.height-pad*2)/graphHeight,1.25);transform.x=(rect.width-graphWidth*transform.k)/2;transform.y=(rect.height-graphHeight*transform.k)/2;applyTransform();}
        function zoom(factor,cx=viewport.clientWidth/2,cy=viewport.clientHeight/2){const next=Math.max(.08,Math.min(4,transform.k*factor));transform.x=cx-(cx-transform.x)*next/transform.k;transform.y=cy-(cy-transform.y)*next/transform.k;transform.k=next;applyTransform();}
        function exportSvg(){const clone=svg.cloneNode(true);clone.setAttribute('xmlns',NS);clone.setAttribute('width',viewport.clientWidth);clone.setAttribute('height',viewport.clientHeight);const style=document.createElementNS(NS,'style');style.textContent='.cluster-halo{fill:transparent;stroke:none}.edge{fill:none;stroke-width:1.45;opacity:.65}.node.source circle{stroke:rgba(255,255,255,.82);stroke-width:2}.node.target circle{stroke:#fff;stroke-width:4}.node text{font-family:Arial,sans-serif;text-anchor:middle}.node .name{fill:#fff;font-size:13px;font-weight:700}.node .params{fill:rgba(255,255,255,.92);font:9.5px monospace}.node .description{fill:rgba(255,255,255,.78);font-size:9.5px}';clone.insertBefore(style,clone.firstChild);const blob=new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'});downloadBlob(blob,'__DATASET___tool_merge_graph.svg');}
        function exportPng(){const clone=svg.cloneNode(true);clone.setAttribute('xmlns',NS);clone.setAttribute('width',viewport.clientWidth);clone.setAttribute('height',viewport.clientHeight);const data=new XMLSerializer().serializeToString(clone),img=new Image(),url=URL.createObjectURL(new Blob([data],{type:'image/svg+xml'}));img.onload=()=>{const scale=2,canvas=document.createElement('canvas');canvas.width=viewport.clientWidth*scale;canvas.height=viewport.clientHeight*scale;const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.fillStyle='#f3f7f4';ctx.fillRect(0,0,viewport.clientWidth,viewport.clientHeight);ctx.drawImage(img,0,0);canvas.toBlob(blob=>{if(blob)downloadBlob(blob,'__DATASET___tool_merge_graph.png');URL.revokeObjectURL(url);});};img.src=url;}
        function downloadBlob(blob,name){const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}
        let panning=false,lastPan=null;
        svg.addEventListener('pointerdown',event=>{if(event.target.closest('.node'))return;panning=true;lastPan={x:event.clientX,y:event.clientY};svg.classList.add('panning');svg.setPointerCapture(event.pointerId);});
        svg.addEventListener('pointermove',event=>{if(!panning)return;transform.x+=event.clientX-lastPan.x;transform.y+=event.clientY-lastPan.y;lastPan={x:event.clientX,y:event.clientY};applyTransform();});
        svg.addEventListener('pointerup',()=>{panning=false;svg.classList.remove('panning');});
        svg.addEventListener('wheel',event=>{event.preventDefault();const rect=viewport.getBoundingClientRect();zoom(event.deltaY<0?1.14:.88,event.clientX-rect.left,event.clientY-rect.top);},{passive:false});
        radiusInput.addEventListener('input',()=>{if(selectedNode){const visual=groupVisuals.find(item=>item.nodes.includes(selectedNode));if(visual&&!resizeNode(selectedNode,Number(radiusInput.value),visual))syncRadiusControl();}});document.getElementById('radius-down').addEventListener('click',()=>{if(selectedNode){const visual=groupVisuals.find(item=>item.nodes.includes(selectedNode));if(visual)resizeNode(selectedNode,selectedNode.width/2-5,visual);}});document.getElementById('radius-up').addEventListener('click',()=>{if(selectedNode){const visual=groupVisuals.find(item=>item.nodes.includes(selectedNode));if(visual)resizeNode(selectedNode,selectedNode.width/2+5,visual);}});search.addEventListener('input',render);mergedOnly.addEventListener('change',render);document.getElementById('fit').addEventListener('click',fitGraph);document.getElementById('zoom-in').addEventListener('click',()=>zoom(1.25));document.getElementById('zoom-out').addEventListener('click',()=>zoom(.8));document.getElementById('export-svg').addEventListener('click',exportSvg);document.getElementById('export-png').addEventListener('click',exportPng);document.getElementById('toggle-toolbar').addEventListener('click',event=>{const header=document.querySelector('header'),collapsed=header.classList.toggle('collapsed');event.currentTarget.textContent=collapsed?'Show':'Hide';event.currentTarget.title=collapsed?'Show toolbar':'Hide toolbar';requestAnimationFrame(fitGraph);});window.addEventListener('resize',fitGraph);render();
    </script>
</body>
</html>"""
    return (
        template.replace("__DATASET__", escape(dataset))
        .replace("__INPUT__", str(input_tools))
        .replace("__OUTPUT__", str(merged_tools))
        .replace("__REDUCTION__", str(reduction))
        .replace("__RATE__", f"{reduction_rate:.1f}")
        .replace("__GRAPH_DATA__", graph_json)
        .replace(
            "__WARNING__",
            (
                f'<span class="warning">{missing_count} unresolved sources</span>'
                if missing_count
                else ""
            ),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize source tools and their merged canonical tools."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--merge-dir",
        required=True,
        help="directory containing clusters.jsonl and merged_tools.json",
    )
    parser.add_argument("--tools", default=None, help="override source tool JSON")
    parser.add_argument("--output", default=None, help="output HTML path")
    args = parser.parse_args()

    merge_dir = Path(args.merge_dir)
    clusters_path = merge_dir / "clusters.jsonl"
    merged_tools_path = merge_dir / "merged_tools.json"
    if not clusters_path.exists():
        raise FileNotFoundError(clusters_path)
    if not merged_tools_path.exists():
        raise FileNotFoundError(merged_tools_path)

    tools_file = resolve_tools_file(args.dataset, args.tools)
    source_tools = load_source_tools(tools_file)
    merged_tools_raw = json.loads(merged_tools_path.read_text(encoding="utf-8"))
    if not isinstance(merged_tools_raw, list):
        raise ValueError(f"Expected a list of tools in {merged_tools_path}")
    groups, missing_count = build_groups(read_jsonl(clusters_path), source_tools)
    output = (
        Path(args.output)
        if args.output
        else merge_dir / "merged_tools_visualization.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_graph_html(
            args.dataset,
            merge_dir,
            groups,
            len(source_tools),
            len(merged_tools_raw),
            missing_count,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "input_tools": len(source_tools),
                "merged_tools": len(merged_tools_raw),
                "display_groups": len(groups),
                "non_singleton_groups": sum(
                    len(group["sources"]) > 1 for group in groups
                ),
                "missing_source_tools": missing_count,
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
