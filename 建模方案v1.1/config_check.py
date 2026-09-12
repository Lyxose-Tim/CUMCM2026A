"""config_check.py —— 加载并校验 A题_config.yaml（类型/单位/来源标签/开关/阈值）。
用法: python config_check.py A题_config.yaml
配置解析通过 ≠ 模型验证通过。"""
import sys, yaml
REQ_TOP=["version","geometry","initial","bc","air","radius","props","numerics","q4","criterion","output","acceptance","production","storage"]
def num(d,path,unit=None,source=True):
    v=d
    for p in path.split('.'): v=v[p]
    assert isinstance(v,dict) and isinstance(v.get("value"),(int,float)), f"{path}: 需 {{value,unit,source}}"
    if unit: assert v.get("unit")==unit, f"{path}: unit 应为 {unit}"
    if source: assert v.get("source") in {"题面Q1","附录2","附录3","附录4","附件1","附件2"} or str(v.get("source","")).startswith(("D","B","H")), f"{path}: source 标签缺失"
    return v["value"]
def check(cfg):
    for k in REQ_TOP: assert k in cfg, f"缺少顶层键 {k}"
    R0=num(cfg,"geometry.R0","m"); L=num(cfg,"geometry.L","m"); assert R0==0.02 and L==0.25
    assert num(cfg,"initial.T0","degC")==28.0 and num(cfg,"initial.C0","kg/kg")==2.55
    assert num(cfg,"bc.h","W/(m^2 K)")==25.0 and num(cfg,"bc.hm","m/s")==8.0e-7
    assert cfg["bc"]["mass_basis"]=="effective_C" and cfg["bc"]["heat_latent"] is False
    a=cfg["air"]; assert a["read_only"] is True and a["interp"] in ("linear","smooth121") and a["extrapolation"] in ("hold_window_mean","hold_last")
    assert a["window_s"]==[10800,14400] and a["breakpoints_s"]==[14400]
    assert cfg["radius"]["after_72h"]=="hold_last" and cfg["radius"]["interp"]=="linear"
    n=cfg["numerics"]; assert n["interface"] in ("harmonic","integral") and n["scheme_baseline"]=="backward_euler"
    assert n["bdf"]["rtol"]<=1e-8 and n["picard"]["max_iter"]>=1 and n["retry"]["dt_min_s"]>0
    assert all(isinstance(x,int) and x>0 for x in n["N_verify"])
    assert cfg["q4"]["kinematics"]=="affine" and cfg["q4"]["frame"]=="reference_x" and cfg["q4"]["mesh_velocity"]=="solid"
    c=cfg["criterion"]; assert c["threshold"]==0.15 and c["domain"]=="full_unrounded"
    o=cfg["output"]; assert o["decimals"]==4 and o["excel_max_rows_including_header"]==1048576
    assert o["result4"]["outside_fill"]=="blank" and o["result4"]["surface_col"]=="药材表面"
    assert o["a1_text"]=="时间\\到药材中心的距离"
    acc=cfg["acceptance"]; assert acc["t_star_h"]==0.02 and acc["table_points"]["dC"]==5e-4
    assert cfg["production"]["approved"] is False or cfg["production"]["config_id"], "approved=true 须给 config_id"
    return True
if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else "A题_config.yaml"
    with open(path,encoding="utf-8") as f: cfg=yaml.safe_load(f)
    check(cfg); print("config OK:", path, "| version", cfg["version"], "| interface", cfg["numerics"]["interface"], "| production.approved", cfg["production"]["approved"])
