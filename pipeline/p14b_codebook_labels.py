# -*- coding: utf-8 -*-
"""Phase 14b. 코드북에서 프롬프트 라벨을 생성한다.

손으로 번역하지 않는다. 프로젝트가 이미 공식 영어본을 갖고 있으므로 거기서 뽑는다.
  data/DATAMAP_EN (2).xlsx                     문항별 보기 라벨 (1)Male (2)Female ...
  data/somting else +@ (1)/10_translation_dictionary.csv   한글 값 -> 영어 값
  data/mode_los_en .../Block01_BASIC_..._EN.xlsx           설문에서 실제로 표시된 표기

왜 필요한가. 이전 프롬프트는 코드를 그대로 찍어서 "income band 3", "2 car(s)" 처럼
사람이 못 알아듣는 문장을 내보냈다. 게다가 D6 은 대수가 아니라 구간 코드라서
코드 2 를 "2 car(s)" 로 쓰면 **사실이 틀린다** (코드 2 = 1대).

출력
  configs/trc_codebook_labels.yaml
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from common import CONFIG, DATA, ROOT

DATAMAP_EN = DATA / "DATAMAP_EN (2).xlsx"
TRANS = DATA / "somting else +@ (1)" / "10_translation_dictionary.csv"
DATA_KO = DATA / "DATA_평일 통행·업무 통행 설문조사_260624 (2).xlsx"
DATA_ENG = DATA / "DATA_EN (1).xlsx"
SURVEY_PDF = DATA / "survey" / "English_Final_Survey_Block01_BASIC.pdf"
SPTAB = (DATA / "mode_los_en final_excel fime (2)"
         / "Block01_BASIC_SP_tables_12tables_pdfdisplay_noq_EN.xlsx")
OUT = CONFIG / "trc_codebook_labels.yaml"

# 프롬프트에 쓰는 문항. DGP 가 쓰는 것(SQ2 D4_1 D4_2 D5 D6 D7 D8)은 반드시 포함한다.
WANT = ["SQ2", "QQ1", "QQ3", "QQ4", "QQ5", "QQ6", "QQ7",
        "D1", "D2", "D3", "D5", "D6", "D7", "D8", "D9"]

# 설문지(p11)의 정의를 그대로 옮긴다. 이 문장이 빠지면 응답자가 본 것과 다른 정보가 된다.
FRAMING_NOTE = (
    "This is not a crash probability. It describes whether the dispatched service is likely "
    "to complete the trip normally, without a major service interruption such as dispatch "
    "delay, route change, remote intervention, or a system check.")

SURVEY_NOTE = (
    "The survey was conducted in the Seoul Capital Area (Seoul, Gyeonggi, or Incheon) and "
    "concerns one ordinary weekday-morning commute, school, or work trip. All times and "
    "costs refer to a single one-way trip.")


def parse_levels(s: str) -> dict:
    """'(1)Male\\n(2)Female' -> {1: 'Male', 2: 'Female'}

    보기 안에도 괄호가 있다("Car travel (non-AV)", "Personal mobility (PM)"). 여는 괄호를
    구분자로 삼으면 그 뒤가 잘려 뜻이 달라지므로, 번호 표지 `(n)` 로 잘라 짝을 짓는다.
    """
    parts = re.split(r"\((\d+)\)", str(s))
    out = {}
    for i in range(1, len(parts) - 1, 2):
        lab = parts[i + 1].strip().strip("/").strip()
        if lab and lab.lower() != "nan":
            out[int(parts[i])] = " ".join(lab.split())
    return out


def run() -> dict:
    print("\n=== Phase 14b. 코드북 라벨 생성 ===")
    for p in (DATAMAP_EN, TRANS, SPTAB):
        if not p.exists():
            raise SystemExit("원본 없음: %s" % p)

    d = pd.read_excel(DATAMAP_EN, sheet_name="map", header=0)
    d.columns = ["var", "desc", "levels", "note"]
    d["var"] = d["var"].astype(str).str.strip()

    values, missing = {}, []
    for v in WANT:
        r = d[d["var"] == v]
        lv = parse_levels(r.iloc[0]["levels"]) if len(r) else {}
        if not lv:
            missing.append(v)
        values[v] = lv
    if missing:
        raise SystemExit("코드북에서 보기를 못 읽은 문항: %s" % missing)

    # 혼잡도 — 설문에서 실제 표시된 표기를 쓴다. 번역사전은 '보통'을 Neutral 로 두지만
    # 응답자에게 보인 표는 Moderate 였다. 표시본을 따른다.
    sp = pd.read_excel(SPTAB, sheet_name="SP tables", header=None)
    shown = set()
    for i, row in sp.iterrows():
        if str(row[0]).strip() == "Crowding":
            shown |= {str(x).strip() for x in row[1:] if pd.notna(x) and str(x).strip() != "-"}
    tr = pd.read_csv(TRANS)
    kmap = dict(zip(tr.korean_label.astype(str).str.strip(), tr.english_translation))
    crowd = {}
    for ko, en_dict in (("여유", "Light"), ("보통", "Moderate"), ("혼잡", "Crowded")):
        en = en_dict if en_dict in shown else kmap.get(ko, en_dict)
        crowd[ko] = en
    print("  혼잡도 표기: %s  (설문 표시본에서 확인된 값 %s)" % (crowd, sorted(shown)))

    # AV 프레이밍 — 번역사전에 없다. 설문 표의 행 이름과 문항 문구에서 가져온다.
    framing = {"자율주행차 작동 안정성": "Operation stability",
               "서비스 이상 가능성": "Service abnormality"}

    # 지역명 — 프롬프트가 영어인데 SQ1_1/SQ1_2 는 한글이다. 로마자 표기가 DATA_EN 에 있고
    # 두 파일이 IDX 로 1:1 대응하므로 거기서 뽑는다. 손으로 옮기면 77개 시군구에서 틀린다.
    regions = {}
    if DATA_KO.exists() and DATA_ENG.exists():
        ko = pd.read_excel(DATA_KO)[["IDX", "SQ1_1", "SQ1_2"]]
        en = pd.read_excel(DATA_ENG)[["IDX", "SQ1_1", "SQ1_2"]]
        j = ko.merge(en, on="IDX", suffixes=("_ko", "_en"))
        for a, b in (("SQ1_1_ko", "SQ1_1_en"), ("SQ1_2_ko", "SQ1_2_en")):
            bad = int((j.groupby(a)[b].nunique() > 1).sum())
            if bad:
                raise SystemExit("지역명 대응이 1:1 이 아니다 (%s 에서 %d건)" % (a, bad))
            regions.update(dict(j.drop_duplicates([a])[[a, b]].values))
        print("  지역명 %d개 (시도 3 + 시군구 %d)" % (len(regions), len(regions) - 3))
    else:
        print("  [warn] DATA/DATA_EN 을 못 찾아 지역명은 한글로 남는다")

    # 거리대별 상황 설명 — 설문지 p12-p17 의 문항 도입부다. 응답자는 표만 본 것이 아니라
    # "동네 안 가까운 목적지", "서울 안 떨어진 업무지구" 같은 **상황 설명**을 함께 봤다.
    # 이게 없으면 LLM 은 숫자만 보고 판단하게 되어 응답자와 다른 정보로 답하는 셈이 된다.
    bands, dist = {}, {}
    if SURVEY_PDF.exists():
        from pypdf import PdfReader
        rd = PdfReader(str(SURVEY_PDF))
        for i in range(len(rd.pages)):
            t = rd.pages[i].extract_text() or ""
            m = re.search(r"Question set \d+ of 6 - (D\d) ([^\n]+?)\(about ([\d.]+) km\)\n(.*?)"
                          r"(?=\n\s*2026 current condition)", t, re.S)
            if not m:
                continue
            code, title, km, body = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
            body = " ".join(body.split())
            # 공통 안내문("There is no correct answer ...")은 system 프롬프트에 이미 있으므로 뺀다.
            body = re.split(r"(?:If the conditions in the table|Compare the time|Do not choose only"
                            r"|Consider time,|The values in the table|There is no correct answer)",
                            body)[0].strip()
            # "For example, this could be a nearby workplace, school, public office, ..." 같은
            # 열거는 상황을 좁히지 못하면서 토큰만 먹는다. 첫 문장(상황 정의)만 남긴다.
            body = re.split(r"(?<=\.)\s+(?:For example|The trip may|Time, cost)", body)[0].strip()
            bands[code] = body
            dist[code] = float(km)
        print("  거리대 상황설명 %d개 / 대표거리 %s" % (len(bands), dist))
    else:
        print("  [warn] 설문지 PDF 없음 — 거리대 상황설명을 넣지 못한다")

    doc = {
        "band_situation": bands,
        "band_distance_km": dist,
        "_source": {
            "codebook": DATAMAP_EN.name,
            "translation_dictionary": str(TRANS.relative_to(DATA)),
            "displayed_tables": SPTAB.name,
            "survey_pdf": str(SURVEY_PDF.name) + " (p11-p17)",
        },
        "survey_note": SURVEY_NOTE,
        "framing_note": FRAMING_NOTE,
        "modes": {
            "PT": "Public transit",
            "Car": "Car travel (non-AV)",
            "PM": "Personal mobility (PM)",
            "Walk": "Walking",
            "AV": "Driverless AV ride-hailing",
        },
        "regions": regions,
        "crowding": crowd,
        "framing_attribute": framing,
        "values": values,
    }
    OUT.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for v in WANT:
        print("  %-5s %s" % (v, "; ".join("%d=%s" % (k, x) for k, x in values[v].items())[:96]))
    print("  [ok] %s" % OUT.relative_to(ROOT))
    return doc


if __name__ == "__main__":
    run()
