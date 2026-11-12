"""Create a self-contained HTML view of a clustered weighted tool graph."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

PALETTE = [
    "#2563eb",
    "#e11d48",
    "#16a34a",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#db2777",
    "#65a30d",
    "#9333ea",
    "#0f766e",
    "#c2410c",
    "#4f46e5",
    "#be123c",
    "#15803d",
    "#a16207",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(components: Any, knowledge_graph: Any) -> dict[str, Any]:
    if not isinstance(components, list):
        raise ValueError("connected_components.json must contain a list")
    if not isinstance(knowledge_graph, Mapping):
        raise ValueError("knowledge_graph.json must contain an object")

    nodes: dict[str, dict[str, Any]] = {}
    cluster_by_uid: dict[str, str] = {}
    clusters: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            continue
        cluster_id = str(component.get("component_id") or f"component:{index:05d}")
        members = component.get("members", [])
        if not isinstance(members, list):
            raise ValueError(f"members for {cluster_id} must be a list")
        member_uids: list[str] = []
        for member in members:
            if not isinstance(member, Mapping):
                continue
            uid = str(member.get("uid", "")).strip()
            if not uid:
                continue
            member_uids.append(uid)
            nodes[uid] = {
                "id": uid,
                "name": str(member.get("name") or uid),
                "cluster": cluster_id,
            }
            cluster_by_uid[uid] = cluster_id
        clusters.append({"id": cluster_id, "members": member_uids})

    raw_nodes = knowledge_graph.get("nodes_detail", [])
    if isinstance(raw_nodes, list):
        for raw_node in raw_nodes:
            if not isinstance(raw_node, Mapping):
                continue
            uid = str(raw_node.get("uid", "")).strip()
            if uid in nodes and raw_node.get("name"):
                nodes[uid]["name"] = str(raw_node["name"])

    raw_edges = knowledge_graph.get("edges_detail", knowledge_graph.get("edges", []))
    if not isinstance(raw_edges, list):
        raise ValueError("knowledge_graph.json edges must be a list")
    edges: list[dict[str, Any]] = []
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, Mapping):
            continue
        source = str(raw_edge.get("source") or raw_edge.get("src") or "").strip()
        target = str(raw_edge.get("target") or raw_edge.get("dst") or "").strip()
        if source not in nodes or target not in nodes or source == target:
            continue
        weight = raw_edge.get("weight", raw_edge.get("similarity", 0))
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            numeric_weight = 0.0
        edges.append(
            {
                "id": f"edge-{index}",
                "source": source,
                "target": target,
                "weight": numeric_weight,
                "internal": cluster_by_uid[source] == cluster_by_uid[target],
            }
        )

    for index, cluster in enumerate(clusters):
        cluster["color"] = PALETTE[index % len(PALETTE)]
        cluster["size"] = len(cluster["members"])
    return {"nodes": list(nodes.values()), "edges": edges, "clusters": clusters}


def render_html(title: str, graph: Mapping[str, Any]) -> str:
    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    safe_title = html.escape(title, quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
* {{ box-sizing: border-box; }} body {{ margin: 0; background: #f5f7fb; color: #172033; overflow: hidden; }}
header {{ height: 74px; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; background: #fff; border-bottom: 1px solid #dce2ec; }}
h1 {{ margin: 0; font-size: 21px; font-weight: 700; }} .stats {{ color: #536078; font-size: 13px; }}
#viewport {{ height: calc(100vh - 74px); position: relative; overflow: hidden; cursor: grab; }} #viewport.dragging {{ cursor: grabbing; }}
svg {{ width: 100%; height: 100%; display: block; }} .edge {{ stroke-linecap: round; }} .node {{ cursor: pointer; }}
.node circle {{ stroke: #fff; stroke-width: 1.5; }} .node {{ cursor: move; }} .cluster-halo {{ fill-opacity: .035; stroke-width: 2; stroke-dasharray: 7 6; cursor: move; }}
.legend {{ position: absolute; top: 18px; right: 18px; max-height: calc(100% - 36px); overflow: auto; width: 245px; padding: 13px 15px; background: rgba(255,255,255,.94); border: 1px solid #dce2ec; border-radius: 10px; box-shadow: 0 8px 24px rgba(30,41,59,.08); }}
.legend-title {{ font-weight: 700; font-size: 13px; margin-bottom: 9px; }} .item {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 12px; cursor: pointer; opacity: .78; }} .item:hover,.item.active {{ opacity: 1; font-weight: 700; }} .swatch {{ width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto; }}
.toggle {{ display: flex; align-items: center; gap: 7px; margin: 0 0 11px; padding-bottom: 10px; border-bottom: 1px solid #e5e9f0; color: #536078; font-size: 12px; cursor: pointer; }} .toggle input {{ accent-color: #2563eb; }}
.hint {{ position: absolute; left: 18px; bottom: 16px; padding: 8px 11px; color: #536078; font-size: 12px; background: rgba(255,255,255,.9); border: 1px solid #dce2ec; border-radius: 7px; }}
</style></head><body><header><h1>{safe_title}</h1><div class="stats" id="stats"></div></header><div id="viewport"><svg id="graph" role="img" aria-label="Clustered tool graph"></svg><div class="legend"><div class="legend-title">Clusters</div><label class="toggle"><input id="show-singletons" type="checkbox"> Show single-element clusters</label><div id="legend-items"></div></div><div class="hint">Drag to pan · Scroll to zoom</div></div>
<script>
const DATA = {payload}; const svg = document.getElementById('graph'); const NS='http://www.w3.org/2000/svg';
const cluster = Object.fromEntries(DATA.clusters.map(c=>[c.id,c])); const nodeById=Object.fromEntries(DATA.nodes.map(n=>[n.id,n]));
document.getElementById('stats').textContent = `${{DATA.nodes.length}} tools · ${{DATA.edges.length}} edges · ${{DATA.clusters.length}} clusters`;
document.getElementById('legend-items').innerHTML=DATA.clusters.map(c=>`<div class="item" data-cluster="${{c.id}}" data-singleton="${{c.size===1}}"><span class="swatch" style="background:${{c.color}}"></span><span>${{c.id}} · ${{c.size}} tools</span></div>`).join('');
const positions={{}}, groups={{}}, nodeVisuals={{}}, edgeVisuals=[]; const width=()=>svg.clientWidth, height=()=>svg.clientHeight;
const columns=Math.max(1,Math.ceil(Math.sqrt(DATA.clusters.length*1.35))), gapX=300, gapY=245;
DATA.clusters.forEach((c,i)=>{{const col=i%columns,row=Math.floor(i/columns); const cx=150+col*gapX,cy=140+row*gapY; const radius=Math.max(36,Math.min(112,24+Math.sqrt(c.members.length)*13)); groups[c.id]={{cx,cy,radius}}; c.members.forEach((id,j)=>{{const angle=j/Math.max(c.members.length,1)*Math.PI*2; const r=c.members.length===1?0:Math.min(radius-10,18+Math.sqrt(c.members.length)*8); positions[id]={{x:cx+Math.cos(angle)*r,y:cy+Math.sin(angle)*r}};}});}});
const worldWidth=Math.max(gapX,columns*gapX), worldHeight=Math.max(gapY,Math.ceil(DATA.clusters.length/columns)*gapY); const content=document.createElementNS(NS,'g'); svg.appendChild(content); const haloLayer=document.createElementNS(NS,'g'), edgeLayer=document.createElementNS(NS,'g'), nodeLayer=document.createElementNS(NS,'g'); content.append(haloLayer,edgeLayer,nodeLayer);
const minWeight=Math.min(...DATA.edges.map(e=>e.weight),1), maxWeight=Math.max(...DATA.edges.map(e=>e.weight),1); const updateEdge=(visual,e)=>{{const a=positions[e.source],b=positions[e.target];visual.setAttribute('x1',a.x);visual.setAttribute('y1',a.y);visual.setAttribute('x2',b.x);visual.setAttribute('y2',b.y);}};
DATA.edges.forEach(e=>{{const line=document.createElementNS(NS,'line'); line.setAttribute('class','edge');line.setAttribute('stroke',e.internal?cluster[nodeById[e.source].cluster].color:'#111827');line.setAttribute('stroke-opacity',e.internal?'.38':'.8');line.setAttribute('stroke-width',e.internal?String(1+Math.max(0,(e.weight-minWeight)/(maxWeight-minWeight||1))*2):'1.4');edgeLayer.appendChild(line);edgeVisuals.push({{element:line,data:e}});updateEdge(line,e);}});
const updateCluster=(clusterId,dx,dy)=>{{const group=groups[clusterId];group.cx+=dx;group.cy+=dy;DATA.clusters.find(c=>c.id===clusterId).members.forEach(id=>{{positions[id].x+=dx;positions[id].y+=dy;nodeVisuals[id].setAttribute('transform',`translate(${{positions[id].x}},${{positions[id].y}})`);}});const halo=haloVisuals[clusterId];halo.setAttribute('cx',group.cx);halo.setAttribute('cy',group.cy);edgeVisuals.forEach(v=>updateEdge(v.element,v.data));}};
const haloVisuals={{}}; DATA.clusters.forEach(c=>{{const halo=document.createElementNS(NS,'circle');const group=groups[c.id];halo.setAttribute('class','cluster-halo');halo.dataset.cluster=c.id;halo.setAttribute('cx',group.cx);halo.setAttribute('cy',group.cy);halo.setAttribute('r',group.radius);halo.setAttribute('stroke',c.color);halo.setAttribute('fill',c.color);haloLayer.appendChild(halo);haloVisuals[c.id]=halo;let moving=false,last={{x:0,y:0}};halo.addEventListener('pointerdown',event=>{{event.stopPropagation();moving=true;last={{x:event.clientX,y:event.clientY}};halo.setPointerCapture(event.pointerId);}});halo.addEventListener('pointermove',event=>{{if(!moving)return;const dx=(event.clientX-last.x)/scale,dy=(event.clientY-last.y)/scale;last={{x:event.clientX,y:event.clientY}};updateCluster(c.id,dx,dy);}});halo.addEventListener('pointerup',()=>moving=false);}});
DATA.nodes.forEach(n=>{{const g=document.createElementNS(NS,'g');const p=positions[n.id];g.setAttribute('class','node');g.dataset.cluster=n.cluster;g.setAttribute('transform',`translate(${{p.x}},${{p.y}})`);const circle=document.createElementNS(NS,'circle');circle.setAttribute('r','6');circle.setAttribute('fill',cluster[n.cluster].color);g.appendChild(circle);nodeLayer.appendChild(g);nodeVisuals[n.id]=g;let moving=false,last={{x:0,y:0}};g.addEventListener('pointerdown',event=>{{event.stopPropagation();moving=true;last={{x:event.clientX,y:event.clientY}};g.setPointerCapture(event.pointerId);circle.setAttribute('r','9');}});g.addEventListener('pointermove',event=>{{if(!moving)return;const dx=(event.clientX-last.x)/scale,dy=(event.clientY-last.y)/scale;last={{x:event.clientX,y:event.clientY}};positions[n.id].x+=dx;positions[n.id].y+=dy;g.setAttribute('transform',`translate(${{positions[n.id].x}},${{positions[n.id].y}})`);edgeVisuals.forEach(v=>updateEdge(v.element,v.data));}});g.addEventListener('pointerup',()=>{{moving=false;circle.setAttribute('r','6');}});}});
let scale=1,tx=0,ty=0,drag=false,last={{x:0,y:0}}; function transform(){{content.setAttribute('transform',`translate(${{tx}},${{ty}}) scale(${{scale}})`);}} function fit(){{scale=Math.min((width()-280)/worldWidth,height()/worldHeight)*.92;scale=Math.max(.18,Math.min(1.5,scale));tx=20;ty=10;transform();}} fit(); svg.addEventListener('wheel',e=>{{e.preventDefault();const rect=svg.getBoundingClientRect(),old=scale;scale=Math.max(.12,Math.min(3,scale*(e.deltaY<0?1.12:.89)));const px=e.clientX-rect.left,py=e.clientY-rect.top;tx=px-(px-tx)*(scale/old);ty=py-(py-ty)*(scale/old);transform();}},{{passive:false}}); svg.addEventListener('pointerdown',e=>{{if(e.target!==svg)return;drag=true;last={{x:e.clientX,y:e.clientY}};svg.setPointerCapture(e.pointerId);document.getElementById('viewport').classList.add('dragging');}});svg.addEventListener('pointermove',e=>{{if(!drag)return;tx+=e.clientX-last.x;ty+=e.clientY-last.y;last={{x:e.clientX,y:e.clientY}};transform();}});svg.addEventListener('pointerup',()=>{{drag=false;document.getElementById('viewport').classList.remove('dragging');}});
const showSingletons=document.getElementById('show-singletons'); const refreshVisibility=()=>{{const visible=showSingletons.checked;DATA.clusters.forEach(c=>{{const display=visible||c.size>1?'':'none';haloVisuals[c.id].style.display=display;document.querySelector(`.item[data-cluster="${{c.id}}"]`).style.display=display;}});DATA.nodes.forEach(n=>{{nodeVisuals[n.id].style.display=visible||cluster[n.cluster].size>1?'':'none';}});edgeVisuals.forEach(v=>{{const sourceCluster=cluster[nodeById[v.data.source].cluster],targetCluster=cluster[nodeById[v.data.target].cluster];v.element.style.display=visible||sourceCluster.size>1&&targetCluster.size>1?'':'none';}});}}; showSingletons.addEventListener('change',refreshVisibility); refreshVisibility();
document.querySelectorAll('.item').forEach(item=>item.addEventListener('click',()=>{{const id=item.dataset.cluster;const active=item.classList.toggle('active');document.querySelectorAll('.item').forEach(other=>{{if(other!==item)other.classList.remove('active');}});content.classList.toggle('has-selection',active);DATA.nodes.forEach(n=>{{nodeVisuals[n.id].style.opacity=!active||n.cluster===id?'1':'.12';}});edgeVisuals.forEach(v=>{{const selected=nodeById[v.data.source].cluster===id||nodeById[v.data.target].cluster===id;v.element.style.opacity=!active||selected?'':'.06';}});}}));
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize entropy clusters as an HTML graph."
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        type=Path,
        help="Directory containing connected_components.json and knowledge_graph.json",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--title", default=None, help="HTML title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    components_path = args.graph_dir / "connected_components.json"
    knowledge_path = args.graph_dir / "knowledge_graph.json"
    if not components_path.is_file():
        raise FileNotFoundError(components_path)
    if not knowledge_path.is_file():
        raise FileNotFoundError(knowledge_path)
    graph = build_graph(load_json(components_path), load_json(knowledge_path))
    output = args.output or args.graph_dir / "cluster_graph_visualization.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_html(args.title or args.graph_dir.name, graph), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "clusters": len(graph["clusters"]),
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
