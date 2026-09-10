from pathlib import Path
import math
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .env import TacticalFootballEnv
from .actions import ACTION_TABLE
from .config import FORMATIONS, LINE_HEIGHTS, PRESSING, TEMPOS

app=FastAPI(title="Tactical AI",version="0.3.0")
UI_DIR=Path(__file__).resolve().parent/"ui"
app.mount("/static",StaticFiles(directory=str(UI_DIR)),name="static")
env=TacticalFootballEnv(observation_mode="inference",backend="toy")
obs,info=env.reset(seed=1); decision_log=["0' match started"]; history=[]

class TacticsRequest(BaseModel):
    formation:str; pressing:str; line_height:str="mid"; tempo:str="balanced"; substitute:bool=False; advance:bool=True

FORMATION_COORDS={
"4-3-3":[(-.9,0),(-.62,-.34),(-.68,-.12),(-.68,.12),(-.62,.34),(-.28,-.22),(-.34,0),(-.28,.22),(.18,-.32),(.4,0),(.18,.32)],
"4-2-3-1":[(-.9,0),(-.62,-.34),(-.68,-.12),(-.68,.12),(-.62,.34),(-.34,-.14),(-.34,.14),(.02,-.3),(.06,0),(.02,.3),(.4,0)],
"3-5-2":[(-.9,0),(-.66,-.22),(-.72,0),(-.66,.22),(-.25,-.38),(-.3,-.16),(-.34,0),(-.3,.16),(-.25,.38),(.3,-.12),(.3,.12)]}

def find_action(r):
    try: f,p,l,t=FORMATIONS.index(r.formation),PRESSING.index(r.pressing),LINE_HEIGHTS.index(r.line_height),TEMPOS.index(r.tempo)
    except ValueError as e: raise HTTPException(400,str(e))
    for i,a in enumerate(ACTION_TABLE):
        if (a.formation,a.pressing,a.line_height,a.tempo,bool(a.substitute))==(f,p,l,t,bool(r.substitute)):
            if not env.action_masks()[i]: raise HTTPException(400,"Illegal action")
            return i
    raise HTTPException(400,"No matching action")

def team(coords,side,line,press,territory):
    shift={"low":-.08,"mid":0,"high":.1}[line]+{"low":-.03,"medium":.02,"high":.08}[press]+(territory-.5)*.15
    out=[]
    for i,(x,y) in enumerate(coords):
        x=x if i==0 else max(-.92,min(.88,x+shift))
        if side=="right": x,y=-x,-y
        out.append({"x":round(x,4),"y":round(y,4)})
    return out

def pitch():
    s=env.backend.state; tac=info["tactics"]
    left=team(FORMATION_COORDS[tac["formation"]],"left",tac["line_height"],tac["pressing"],s.territory)
    style=info["opponent_style_estimate"]; line="high" if style=="high_press" else ("low" if style=="counterattack" else "mid"); press="high" if style=="high_press" else ("low" if style=="counterattack" else "medium")
    right=team(FORMATION_COORDS["4-3-3"],"right",line,press,1-s.territory)
    ball={"x":round(max(-.94,min(.94,(s.territory-.5)*1.25+.13*math.sin(s.minute*.55))),4),"y":round(.34*math.sin(s.minute*.31+s.possession*3),4)}
    return {"left":left,"right":right,"ball":ball}

def state():
    s=env.backend.state
    return {"minute":s.minute,"score":[s.goals_for,s.goals_against],"tactics":info["tactics"],"opponent_belief":info["opponent_belief"],"opponent_style_estimate":info["opponent_style_estimate"],"substitutions_used":info["substitutions_used"],"max_substitutions":env.config.max_substitutions,"stats":{"possession":[round(s.possession*100),round((1-s.possession)*100)],"shots":[max(s.goals_for,round(s.xg_for*4.8)),max(s.goals_against,round(s.xg_against*4.8))],"xg":[round(s.xg_for,2),round(s.xg_against,2)],"fouls":[int(s.minute/22),int(s.minute/24)],"fatigue":round(s.fatigue*100),"territory":round(s.territory*100)},"pitch":pitch(),"decisions":decision_log[-8:],"terminated":s.minute>=90,"history_length":len(history),"backend":env.backend_name}

def snapshot(): history.append(state())
snapshot()

@app.get("/")
def index(): return FileResponse(UI_DIR/"index.html")
@app.get("/ui-state")
def ui_state(): return state()
@app.get("/history/{index}")
def hist(index:int):
    if index<0 or index>=len(history): raise HTTPException(404,"History index out of range")
    return history[index]
@app.post("/new-match")
def new_match(seed:int=1):
    global obs,info
    obs,info=env.reset(seed=seed); decision_log.clear(); decision_log.append("0' match started"); history.clear(); snapshot(); return state()
@app.post("/apply-tactics")
def apply_tactics(r:TacticsRequest):
    global obs,info
    before=dict(info["tactics"]); idx=find_action(r)
    obs,reward,terminated,truncated,info=env.step(idx)
    changes=[f"{k}: {info['tactics'][k]}" for k in ("formation","pressing","line_height","tempo") if before[k]!=info["tactics"][k]]
    if r.substitute: changes.append("substitution")
    decision_log.append(f"{info['minute']}' "+(", ".join(changes) if changes else "hold tactics")); snapshot()
    return {**state(),"reward":reward,"terminated":terminated,"truncated":truncated}
