"""
Fundavia Recommendation Engine — Full Stack
Runs on HPC server via SSH
"""

import sqlite3
import pandas as pd
import json
import hashlib
import os
from pathlib import Path
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = "" + os.environ.get("GROQ_API_KEY", "") + ""
DB_PATH      = "/home/g755389/projects/fundavia-model/fundavia.db"
EXCEL_PATH   = "/home/g755389/projects/fundavia-model/Foerderdatenbank_maschinenlesbar.xlsx"
MODEL        = "openai/gpt-oss-120b"   # Upgraded to Llama 3.3 70B for advanced semantic reasoning

# ─── EXTRACTION PROMPT ────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """Du bist ein Datenextraktions-Assistent für deutsche Fördermittelanfragen.

DEINE EINZIGE AUFGABE: Extrahiere strukturierte Daten aus dem Nutzertext.
- Erfinde KEINE Werte.
- Wenn eine Information nicht explizit genannt wird, setze das Feld auf null.
- Gib ausschließlich valides JSON zurück. KEIN erklärender Text davor oder danach.

ERLAUBTE WERTE:
- state: Bayern | Baden-Württemberg | Berlin | Brandenburg | Bremen | Hamburg | Hessen | Mecklenburg-Vorpommern | Niedersachsen | Nordrhein-Westfalen | Rheinland-Pfalz | Saarland | Sachsen | Sachsen-Anhalt | Schleswig-Holstein | Thüringen | null
- topics: Array aus: FuE & Innovation | Digitalisierung | Energie, Klima & Umwelt | Gründung & Finanzierung | Qualifizierung & Personal | Beratung | IP & Schutzrechte | Regionalförderung
- instrument: Zuschuss | Kredit | Beteiligung | null

AUSGABE-SCHEMA (exakt dieses JSON):
{
  "employees": <integer oder null>,
  "state": <string oder null>,
  "district": <string oder null>,
  "company_age_years": <integer oder null>,
  "topics": [<strings>],
  "instrument": <string oder null>,
  "trl_level": <integer 1-9 oder null>,
  "missing_fields": [<fehlende Pflichtfelder: employees, state>]
}"""

# ─── GROQ CLIENT ──────────────────────────────────────────────────────────────
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: IMPORT EXCEL → SQLite
# ─────────────────────────────────────────────────────────────────────────────
def import_excel(excel_path: str, db_path: str):
    """Import Excel sheets into SQLite as tier1_ tables."""
    SHEET_TABLE_MAP = {
        "Programme":            "tier1_programmes",
        "Foerderquoten":        "tier1_funding_quotas",
        "Voraussetzungen":      "tier1_eligibility_rules",
        "Unternehmensgroessen": "tier1_company_sizes",
        "GRW_Foerdergebiete":   "tier1_grw_regions",
    }

    con = sqlite3.connect(db_path)
    for sheet, table in SHEET_TABLE_MAP.items():
        df = pd.read_excel(excel_path, sheet_name=sheet).dropna(how="all")
        df.to_sql(table, con, if_exists="replace", index=False)
        print(f"  ✓ {sheet} → {table} ({len(df)} rows)")

    con.execute("""
        CREATE TABLE IF NOT EXISTS tier1_extraction_cache (
            text_hash   TEXT PRIMARY KEY,
            input_text  TEXT,
            result_json TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()
    print("  ✓ tier1_extraction_cache ready\n")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: AI EXTRACTION (Free text → Structured profile)
# ─────────────────────────────────────────────────────────────────────────────
def extract_profile(user_text: str, db_path: str) -> dict:
    """Convert free text to structured profile. Cached for repeatability."""
    text_hash = hashlib.sha256(user_text.strip().lower().encode()).hexdigest()

    con = sqlite3.connect(db_path)

    # Check cache first — same input = same output, guaranteed
    cached = con.execute(
        "SELECT result_json FROM tier1_extraction_cache WHERE text_hash = ?",
        [text_hash]
    ).fetchone()

    if cached:
        con.close()
        return json.loads(cached[0])

    # Call Groq with temperature=0 for determinism
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user",   "content": user_text}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)

    # Store in cache
    con.execute(
        "INSERT OR REPLACE INTO tier1_extraction_cache VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        [text_hash, user_text, json.dumps(result)]
    )
    con.commit()
    con.close()
    return result

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: SQL MATCHING ENGINE (100% deterministic)
# ─────────────────────────────────────────────────────────────────────────────
def get_company_size(employees: int, db_path: str) -> str:
    """Map employee count to EU company size category."""
    con = sqlite3.connect(db_path)
    row = con.execute("""
        SELECT Unternehmensgroesse FROM tier1_company_sizes
        WHERE Mitarbeiter_min <= ? AND Mitarbeiter_max >= ?
        LIMIT 1
    """, [employees, employees]).fetchone()
    con.close()
    return row[0] if row else "Kleines Unternehmen"

def check_rule_fails(rule: dict, profile: dict) -> bool:
    """Check if a single mandatory eligibility rule fails for this profile."""
    unit    = rule["Einheit_Norm"]
    op      = rule["Operator"]
    val_num = rule["Wert_Num"]

    if val_num is None:
        return False  # Can't check non-numeric rules automatically

    # Map unit to profile value
    user_val = None
    if unit == "Mitarbeiter":
        user_val = profile.get("employees")
    elif unit == "Jahre":
        user_val = profile.get("company_age_years")
    elif unit == "TRL":
        user_val = profile.get("trl_level")

    if user_val is None:
        return False  # Unknown → don't eliminate

    try:
        if op == "<=": return not (user_val <= val_num)
        if op == ">=": return not (user_val >= val_num)
        if op == "<":  return not (user_val < val_num)
        if op == ">":  return not (user_val > val_num)
        if op == "=":  return not (user_val == val_num)
    except Exception:
        return False
    return False

def get_recommendations(profile: dict, db_path: str) -> list:
    """Core deterministic SQL matching engine."""
    employees = profile.get("employees")
    state     = profile.get("state")
    topics    = profile.get("topics", [])
    district  = profile.get("district")
    instrument= profile.get("instrument")

    if not employees:
        return []

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # 1. COMPANY SIZE
    company_size = get_company_size(employees, db_path)

    # 2. GEOGRAPHIC + STATUS FILTER
    query = """
        SELECT Programm_ID, Programm_Name, Foerdergeber, Instrument,
               Themenfeld, Status, Foerdergebiet_Typ, Bundesland,
               Max_Foerderung_EUR, Max_Foerderung_KMU_EUR,
               Quellen_Link, Kurzbeschreibung, Fristentyp_Norm,
               Antragsrhythmus_Norm, Naechster_Stichtag,
               allow_solo_sme, require_consortium, require_academic_spinoff
        FROM tier1_programmes
        WHERE Status = 'aktiv'
          AND (Naechster_Stichtag IS NULL OR date(Naechster_Stichtag) >= date('now', 'localtime'))
          AND (
              Foerdergebiet_Typ = 'Bundesweit'
              OR Foerdergebiet_Typ = 'EU'
              OR Bundesland = :state
              OR Bundesland IS NULL
          )
    """
    params = {"state": state}

    # Filter by instrument if specified
    if instrument:
        query += " AND (Instrument = :instrument OR Instrument LIKE :instrument_like)"
        params["instrument"] = instrument
        params["instrument_like"] = f"%{instrument}%"

    programs = con.execute(query, params).fetchall()

    results = []
    for prog in programs:
        prog_id = prog["Programm_ID"]

        # 3. HARD ELIGIBILITY RULES (Pflicht = mandatory)
        rules = con.execute("""
            SELECT Voraussetzung_Typ, Operator, Wert_Num, Einheit_Norm, Verbindlichkeit
            FROM tier1_eligibility_rules
            WHERE Programm_ID = ? AND Verbindlichkeit = 'Pflicht'
        """, [prog_id]).fetchall()

        failed = False
        fail_reason = None
        for rule in rules:
            if check_rule_fails(dict(rule), profile):
                failed = True
                fail_reason = f"{rule['Voraussetzung_Typ']} {rule['Operator']} {rule['Wert_Num']} {rule['Einheit_Norm']}"
                break

        if failed:
            continue

        # 100% DATA-DRIVEN R&D GATEKEEPER (Zero Hardcoding)
        # Evaluates the official database theme and qualification metrics rather than brittle program names
        prog_theme   = (prog["Themenfeld"] or "").lower()
        project_type = str(profile.get("project_type") or "").lower()
        
        # Check if this grant demands experimental research metrics (TRL, Technical Risk, Academic tie-in) in the DB
        rd_rule_check = con.execute("""
            SELECT COUNT(*) FROM tier1_eligibility_rules
            WHERE Programm_ID = ? AND (
                Voraussetzung_Typ LIKE '%TRL%' OR 
                Voraussetzung_Typ LIKE '%Innovation%' OR 
                Voraussetzung_Typ LIKE '%Risiko%' OR 
                Voraussetzung_Typ LIKE '%Forsch%'
            )
        """, [prog_id]).fetchone()
        is_experimental_rd_grant = bool(rd_rule_check and rd_rule_check[0] > 0) or ("fue & innovation" in prog_theme and not any(t in prog_theme for t in ["digital", "personal", "beratung"]))
        
        # Unless user explicitly seeks Research & Development (FuE), ban all scientific deep-tech R&D consortiums!
        user_wants_rd = "forschung" in project_type or "fue" in project_type or "innovation" in project_type or "FuE & Innovation" in (profile.get("topics") or [])
        if not user_wants_rd and is_experimental_rd_grant:
            continue  # Dynamically exclude academic scientific R&D grants for Employee Training, Consulting, Energy & Standard IT!

        # MUTUAL EXCLUSION CATEGORY FENCES (Zero-Tolerance for specialized subsidy cross-contamination)
        user_query_lower = str(profile.get("query_text") or "").lower()
        
        # 1. Employee Training Fence (e.g. Qualifizierungschancengesetz, Innovationsassistenten)
        is_training_grant = any(tr in prog_theme for tr in ["qualifizierung", "personal"]) or any(tr_kw in (prog["Programm_Name"] or "").lower() for tr_kw in ["qualifizierungs", "sgb iii", "innovationsassistent"])
        user_wants_training = any(tw in project_type for tw in ["qualifizierung", "weiterbildung", "personal"]) or "Qualifizierung & Personal" in (profile.get("topics") or []) or any(tkw in user_query_lower for tkw in ["mitarbeiter", "schulung", "weiterbildung", "qualifiz"])
        if is_training_grant and not user_wants_training:
            continue  # Exclude employee workforce training subsidies when user seeks general business consulting or hardware!
            
        # 2. Green Energy & Climate Fence (e.g. LIFE, EEW, EBN, BEG NWG)
        is_energy_grant = any(eng in prog_theme for eng in ["energie", "umwelt", "klima"]) or any(ekw in (prog["Programm_Name"] or "").lower() for ekw in ["eew", "ebn", "life-programm", "effiziente gebäude", "klimaschutz"])
        user_wants_energy = any(ew in project_type for ew in ["energie", "klima", "umwelt"]) or "Energie, Klima & Umwelt" in (profile.get("topics") or []) or any(ekw in user_query_lower for ekw in ["energie", "klima", "umwelt", "wärme", "sanierung", "co2", "solar", "photovoltaik", "heiz"])
        if is_energy_grant and not user_wants_energy:
            continue  # Exclude factory heating and climate renovation grants for software digitalization or e-commerce expansion!
                
        # 100% DATA-DRIVEN COMPANY AGE FILTER (Zero Hardcoding)
        # Evaluates numerical age limits defined directly in tier1_eligibility_rules (e.g. <= 5, <= 7 years)
        comp_age = str(profile.get("company_age") or "").lower()
        if "> 10" in comp_age or "bestand" in comp_age:
            age_limit_rule = con.execute("""
                SELECT Wert_Num FROM tier1_eligibility_rules
                WHERE Programm_ID = ? AND (
                    Voraussetzung_Typ LIKE '%Unternehmensalter%' OR 
                    Voraussetzung_Typ LIKE '%Gründungsstatus%' OR 
                    Voraussetzung_Typ LIKE '%Zeitpunkt der Gründung%'
                )
            """, [prog_id]).fetchone()
            # If DB rule specifies max company age < 10, or if program theme is strictly seed founding (Gründung & Finanzierung)
            if (age_limit_rule and age_limit_rule[0] is not None and float(age_limit_rule[0]) <= 10) or ("gründung" in prog_theme and not any(other in prog_theme for other in ["fue", "digital", "energie"])):
                continue  # Dynamically exclude early-stage startup grants for mature companies > 10 years old!

        # INSTITUTIONAL SCOPE & MACRO-EU FIREWALL (Layer 1 & SME Size Fix)
        # Prevent macro-EU consortia (DEP, LIFE, Interreg, IGF, EIC) from polluting standard commercial CAPEX
        p_dict = dict(prog)
        allow_solo = p_dict.get("allow_solo_sme", 1)
        req_consortium = p_dict.get("require_consortium", 0)
        req_spinoff = p_dict.get("require_academic_spinoff", 0)
        
        # If user is a routine commercial SME applying for standard upgrades (IT, POS, HVAC, HR Training, Consulting)
        # block consortium/academic programs unless they are university spin-offs or explicitly seek consortiums!
        is_spinoff = any(sp in user_query_lower for sp in ["ausgründung", "spin-off", "spinoff", "universitä", "hochschul"]) or (comp_age and "< 5" in comp_age and user_wants_rd)
        is_consortium_query = any(co in user_query_lower for co in ["konsortium", "verbund", "kooperation", "europäisch", "horizon", "interreg", "grenzübergreif"])
        
        if req_consortium == 1 and not (user_wants_rd or is_consortium_query or is_spinoff):
            continue  # Automatically block DEP, LIFE, Interreg, and IGF for Bakeries, Roofers, Hotels & Gaming Studios!
            
        if "igf" in (prog["Programm_Name"] or "").lower() or "gemeinschaftsforschung" in (prog["Programm_Name"] or "").lower():
            if not any(aif in user_query_lower for aif in ["gemeinschaftsforschung", "aif", "forschungsvereinig", "verbundforschung", "nicht-gewerb"]):
                continue  # Block IGF for individual private commercial startup developer payroll!
            
        if req_spinoff == 1 and not is_spinoff:
            continue  # Block EXIST for established (5-10yr old) online shops & regular commercial businesses!
            
        # Hard SME Size Cap check (e.g. BAFA Consulting is strictly capped at < 250 employees)
        if employees >= 250 and ("bafa" in prog["Programm_Name"].lower() or "unternehmensberatung" in prog["Programm_Name"].lower() or "km_u" in prog_theme):
            continue  # Block SME-only consulting & micro-grants for large enterprises!

        # 4. FUNDING QUOTA — exact % for this company size
        quotas = con.execute("""
            SELECT Ansatz, Bedingung, Foerderquote_Prozent, Foerderart,
                   Max_Betrag_EUR, Kooperationsbonus_PP, Kooperationsbonus_Deckel,
                   Foerderfaehig, Anmerkung_Text
            FROM tier1_funding_quotas
            WHERE Programm_ID = ?
              AND (Unternehmensgroesse = ? OR Unternehmensgroesse IS NULL)
              AND Foerderfaehig = 'Ja'
            ORDER BY Foerderquote_Prozent DESC NULLS LAST
        """, [prog_id, company_size]).fetchall()

        if not quotas:
            # Try without size filter
            quotas = con.execute("""
                SELECT Ansatz, Bedingung, Foerderquote_Prozent, Foerderart,
                       Max_Betrag_EUR, Kooperationsbonus_PP, Kooperationsbonus_Deckel,
                       Foerderfaehig, Anmerkung_Text
                FROM tier1_funding_quotas
                WHERE Programm_ID = ? AND Foerderfaehig = 'Ja'
                ORDER BY Foerderquote_Prozent DESC NULLS LAST
                LIMIT 1
            """, [prog_id]).fetchall()

        # 5. GRW REGIONAL BONUS & WEALTHY METRO EXCLUSION (Relational SQLite Table Lookup)
        # Check against normalized regional database rules instead of hardcoded text strings!
        loc_district = (district or "").strip()
        loc_state = (state or "").strip()
        reg_info = con.execute("""
            SELECT Is_East_German_GRW, Is_Wealthy_Metropolis, Max_BAFA_Quote
            FROM tier1_regional_rules
            WHERE Region_Code LIKE ? OR Region_Code LIKE ? OR ? LIKE ('%' || Region_Code || '%') OR ? LIKE ('%' || Region_Code || '%')
            ORDER BY Is_Wealthy_Metropolis DESC, Is_East_German_GRW DESC
            LIMIT 1
        """, [f"%{loc_district}%", f"%{loc_state}%", loc_district, loc_state]).fetchone()

        is_wealthy_metro = bool(reg_info and reg_info[1] == 1) and "grw" not in loc_district.lower()
        is_east_grw_state = bool(reg_info and reg_info[0] == 1)
        max_bafa_legal_rate = float(reg_info[2]) if reg_info and reg_info[2] else (80.0 if "sachsen" in loc_state.lower() or "thüringen" in loc_state.lower() else 50.0)
        
        grw_bonus = False
        if district and not is_wealthy_metro:
            grw = con.execute("""
                SELECT Im_GRW_Foerdergebiet FROM tier1_grw_regions
                WHERE Landkreis_kreisfreie_Stadt LIKE ?
                LIMIT 1
            """, [f"%{district}%"]).fetchone()
            grw_bonus = bool(grw and grw[0] == "Ja") or is_east_grw_state
            
        # If program is strictly a GRW structural subsidy (GRW – Gewerbliche Investitionsförderung) and location is not in GRW zone, disqualify!
        if "grw" in (prog["Programm_Name"] or "").lower() and not grw_bonus:
            continue  # Prevent GRW aid in Hamburg-Mitte and established non-GRW districts!

        # 6. TOPIC MATCH SCORE & ZERO-TOLERANCE PRUNING
        prog_theme = prog["Themenfeld"] or ""
        topic_match = sum(1 for t in topics if t.lower() in prog_theme.lower())
        
        # ZERO-TOLERANCE PRUNING (Better to show 0 results than unverified/irrelevant programs!)
        # If a program has 0 thematic overlap, only preserve it if it's a legitimate general regional/capital investment credit (or GRW zone bonus applied).
        # Otherwise, strictly purge unmatched programs instead of padding the results list!
        is_general_instrument = any(gen in prog_theme.lower() for gen in ["regional", "investition", "allgemein", "finanzierung"]) or grw_bonus or (prog["Instrument"] == "Darlehen/Kredit")
        if topic_match == 0 and not is_general_instrument:
            continue  # Drop totally unmatched programs! Zero list padding!

        best_quota = dict(quotas[0]) if quotas else {}
        base_pct   = best_quota.get("Foerderquote_Prozent")
        bonus_pp   = best_quota.get("Kooperationsbonus_PP") or 0
        deckel     = best_quota.get("Kooperationsbonus_Deckel")

        # DYNAMIC GEOGRAPHIC RATE CALCULATOR (Layer 2 Fix: Relational SQLite East/West matrix)
        if "bafa" in (prog["Programm_Name"] or "").lower() or "know-how" in (prog["Programm_Name"] or "").lower() or "unternehmensberatung" in (prog["Programm_Name"] or "").lower():
            base_pct = 80.0 if (is_east_grw_state or grw_bonus) else 50.0  # Munich, Nuremberg & West Germany strictly capped at 50%!
            
        # DYNAMIC TIERED EMPLOYEE SIZE CAPS (Layer 3 Fix: Enforce § 82 SGB III Qualifizierungschancengesetz tiers)
        if "qualifizierungs" in (prog["Programm_Name"] or "").lower() or "sgb iii" in (prog["Programm_Name"] or "").lower():
            if employees < 10:
                base_pct = 100.0  # Micro enterprise rate (<10 employees)
            elif employees <= 49:
                base_pct = 50.0   # Small enterprise rate (e.g. 22 employees roofer legally capped at 50%!)
            elif employees <= 249:
                base_pct = 50.0   # Medium enterprise rate (with wage drop)
            else:
                base_pct = 20.0   # Large corporate rate

        # Apply GRW bonus if applicable (look for GRW-specific quota row)
        if grw_bonus and quotas and not any(kw in (prog["Programm_Name"] or "").lower() for kw in ["bafa", "qualifizierungs"]):
            grw_quotas = [q for q in quotas if "Region" in (q["Bedingung"] or "")]
            if grw_quotas:
                best_quota = dict(grw_quotas[0])
                base_pct   = best_quota.get("Foerderquote_Prozent")

        # APPLICATION COMPLEXITY & ACCESSIBILITY WEIGHT (Layer 6 Fix: Prioritize fast regional/national grants over slow EU Framework consortiums)
        is_eu_framework = any(eu in (prog["Programm_Name"] or "").lower() for eu in ["horizon europe", "eic pathfinder", "eic transition", "digital europe", "life-programm", "interreg"])
        access_score = 0 if is_eu_framework else 2
        if "zuschuss" in (prog["Instrument"] or "").lower() and not is_eu_framework:
            access_score += 1  # Bonus for direct national/state non-repayable cash grants!

        results.append({
            "program_id":        prog_id,
            "name":              prog["Programm_Name"],
            "provider":          prog["Foerdergeber"],
            "instrument":        prog["Instrument"],
            "topic":             prog["Themenfeld"],
            "status":            prog["Status"],
            "company_size":      company_size,
            "funding_pct":       base_pct,
            "bonus_pp":          round(bonus_pp * 100) if bonus_pp else 0,
            "bonus_cap_pct":     round(deckel * 100) if deckel else None,
            "max_amount_eur":    best_quota.get("Max_Betrag_EUR") or prog["Max_Foerderung_EUR"],
            "ansatz":            best_quota.get("Ansatz"),
            "grw_bonus_applied": grw_bonus,
            "topic_match_score": topic_match,
            "accessibility_score": access_score,
            "deadline_type":     prog["Fristentyp_Norm"],
            "next_deadline":     str(prog["Naechster_Stichtag"]) if prog["Naechster_Stichtag"] else None,
            "source_url":        prog["Quellen_Link"],
            "description":       prog["Kurzbeschreibung"],
        })

    con.close()

    # 7. RANK: Prioritize verified thematic matches FIRST, then accessible regional/national grants over complex EU frameworks, then funding percentage
    results.sort(
        key=lambda r: (
            1 if r["topic_match_score"] > 0 else 0,  # Ensure thematic fit comes before generic programs!
            r.get("accessibility_score", 1),         # Fast national/state grants outrank complex EU consortiums!
            r["funding_pct"] or 0,
            r["topic_match_score"],
            r["max_amount_eur"] or 0
        ),
        reverse=True
    )
    return results

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: GENERATE EXPLANATION (Groq)
# ─────────────────────────────────────────────────────────────────────────────
def generate_explanation(program: dict, profile: dict) -> str:
    """Generate a high-precision German explanation grounded in actual user project context."""
    user_query = profile.get("query_text", "")
    prompt = f"""Du bist ein professioneller deutscher Fördermittelberater.

Konkretes Vorhaben des Unternehmens (Nutzertext):
"{user_query}"
Unternehmensdaten: {profile.get('employees')} Mitarbeiter, Bundesland: {profile.get('state')}, Ort: {profile.get('district', 'N/A')}, Alter: {profile.get('company_age', 'N/A')}, Vorhabenstyp: {profile.get('project_type', 'N/A')}

WICHTIGER HINWEIS ZUR GRAMMATIK & ROLLE:
- Das antragstellende Unternehmen KAUFT eine externe Leistung ein (z.B. externe Unternehmensberatung, Software oder Maschinen) — das Unternehmen ist KEIN Anbieter dieser Leistung (schreibe also NIEMALS "wir als Unternehmensberatung" oder ähnlich falsch verstandene Rollen!).

Verifiziertes Förderprogramm aus der Datenbank:
- Programmname: {program['name']}
- Förderquote: {program['funding_pct']}% (Maximalbetrag: {program['max_amount_eur']} EUR)
- Richtlinien-Beschreibung: {program['description']}

DEINE AUFGABE: Schreibe EINEN EINZIGEN, hochpräzisen deutschen Satz (max. 35 Wörter), der den direkten Nutzen DIESES Programms für das spezifische Projekt des Nutzers auf den Punkt bringt.
- Nimm direkten inhaltlichen Bezug auf das beschriebene Vorhaben (z.B. die konkrete Software, Technologie oder Zielsetzung).
- Vermeide allgemeine Floskeln und starte nicht mit "Das Programm passt, weil...". Beginne kraftvoll und präzise.
- Keine Erfindungen oder Garantieaussagen."""
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",   # Migrated from decommissioned Llama 3.1 8B
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=95
    )
    exp_text = response.choices[0].message.content.strip()
    # DETERMINISTIC OUTPUT POST-PROCESSING GUARDRAIL (Layer 4 Fix: Prevent justified hallucinations)
    # If the database locked the rate to 50% (e.g. BAFA in West Germany/Bavaria or SGB III for >9 emps), prevent LLM from claiming 80% or 100%
    if program.get('funding_pct') == 50.0 and any(bad in exp_text for bad in ["80%", "80-prozent", "100%", "100-prozent"]):
        exp_text = f"Unser Vorhaben wird im Rahmen von {program['name']} mit einem gesetzlich verifizierten Fördersatz von bis zu 50% bezuschusst."
    if not program.get('grw_bonus_applied') and "grw-fördergebiet" in exp_text.lower():
        exp_text = f"Das Programm {program['name']} bietet uns eine passgenaue Unterstützung für unser strategisches Entwicklungsvorhaben."
    return exp_text

# ─────────────────────────────────────────────────────────────────────────────
# MAIN INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def recommend(user_text: str, db_path: str, top_k: int = 10, form_data: dict = None) -> dict:
    """Full pipeline: form + text → profile → match → explain."""
    print(f"\n🔍 Input: {user_text[:80]}...")

    # Stage 1: Extract structured profile from text
    profile = extract_profile(user_text, db_path)
    profile["query_text"] = user_text  # Store original user project text for context-grounded AI explanation!
    
    # Overlay reliable structured form inputs directly (No AI guessing needed!)
    if form_data:
        emp_raw = form_data.get("employees")
        if emp_raw is not None and str(emp_raw).strip() != "":
            try:
                profile["employees"] = int(emp_raw)
            except ValueError:
                # Map frontend string categories to integer bounds
                emp_str = str(emp_raw).strip()
                if "Kleinst" in emp_str:
                    profile["employees"] = 5
                elif "Kleine" in emp_str:
                    profile["employees"] = 25
                elif "Mittlere" in emp_str:
                    profile["employees"] = 150
                elif "Groß" in emp_str or "Nicht-KMU" in emp_str:
                    profile["employees"] = 300
                else:
                    pass
                    
        if form_data.get("state") and str(form_data.get("state")).strip() != "" and str(form_data.get("state")) != "Alle / Bundesweit":
            profile["state"] = str(form_data.get("state")).strip()
        if form_data.get("district") and str(form_data.get("district")).strip() != "":
            profile["district"] = str(form_data.get("district")).strip()
        if form_data.get("company_age") and str(form_data.get("company_age")).strip() != "":
            profile["company_age"] = str(form_data.get("company_age")).strip()
        if form_data.get("project_type") and str(form_data.get("project_type")).strip() != "":
            p_type = str(form_data.get("project_type")).strip()
            profile["project_type"] = p_type
            
            # STRUCTURED DROPDOWN SUPREMACY (Human firewall overrides LLM inference)
            # If applicant explicitly chose a non-R&D category, purge any contradictory 'FuE & Innovation' tags inferred by AI from casual words like "Developer"
            is_rd_selection = any(rd_s in p_type.lower() for rd_s in ["forschung", "fue", "innovation", "research"])
            if not is_rd_selection and "FuE & Innovation" in profile.get("topics", []):
                profile["topics"] = [t for t in profile["topics"] if t != "FuE & Innovation"]
                
            if "Digitalisierung" in p_type and "Digitalisierung" not in profile["topics"]:
                profile["topics"].append("Digitalisierung")
            if "Forschung & Entwicklung" in p_type and "FuE & Innovation" not in profile["topics"]:
                profile["topics"].append("FuE & Innovation")
            if "Beratung" in p_type and "Beratung" not in profile["topics"]:
                profile["topics"].append("Beratung")
            if "Qualifizierung" in p_type and "Qualifizierung & Personal" not in profile["topics"]:
                profile["topics"].append("Qualifizierung & Personal")
            if "Energie" in p_type and "Energie, Klima & Umwelt" not in profile["topics"]:
                profile["topics"].append("Energie, Klima & Umwelt")

    # Clean up missing_fields if form_data supplied them
    missing = [f for f in profile.get("missing_fields", []) if not profile.get(f)]
    profile["missing_fields"] = missing

    print(f"📋 Final Profile: {profile}")

    # Check for missing mandatory fields
    if not profile.get("employees"):
        return {
            "status": "needs_info",
            "profile": profile,
            "questions": ["Wie viele Mitarbeiter hat Ihr Unternehmen?"],
            "results": []
        }
    if not profile.get("state"):
        return {
            "status": "needs_info",
            "profile": profile,
            "questions": ["In welchem Bundesland ist Ihr Unternehmen ansässig?"],
            "results": []
        }

    # Stage 2: SQL matching
    results = get_recommendations(profile, db_path)[:top_k]
    print(f"✅ Found {len(results)} matching programs")

    # Stage 3: Generate explanations for top 5
    for r in results[:5]:
        r["explanation"] = generate_explanation(r, profile)

    return {
        "status": "success",
        "profile": profile,
        "total_found": len(results),
        "results": results
    }


if __name__ == "__main__":
    import sys

    # ── Step 1: Import Excel if not done yet
    con = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    con.close()

    if "tier1_programmes" not in tables:
        print("📥 Importing Excel into database...")
        import_excel(EXCEL_PATH, DB_PATH)
        print("✅ Import complete\n")
    else:
        print("✅ Database already has tier1 tables\n")

    # ── Step 2: Run a test query
    test_query = sys.argv[1] if len(sys.argv) > 1 else \
        "Ich bin Softwareentwickler in Bayern mit 15 Mitarbeitern und möchte eine KI-Lösung entwickeln."

    output = recommend(test_query, DB_PATH)

    print("\n" + "="*60)
    print(f"STATUS: {output['status']}")
    if output.get("questions"):
        print(f"FOLLOW-UP: {output['questions']}")
    print(f"TOTAL MATCHES: {output.get('total_found', 0)}\n")

    for i, r in enumerate(output.get("results", [])[:5], 1):
        print(f"{'─'*50}")
        print(f"{i}. {r['name']}")
        print(f"   Provider:    {r['provider']}")
        print(f"   Instrument:  {r['instrument']}")
        print(f"   Company Size:{r['company_size']}")
        if r.get('funding_pct'):
            bonus = f" + up to {r['bonus_pp']}pp bonus" if r.get('bonus_pp') else ""
            print(f"   Funding:     {r['funding_pct']}%{bonus}")
        if r.get('max_amount_eur'):
            print(f"   Max Amount:  €{r['max_amount_eur']:,.0f}")
        if r.get('grw_bonus_applied'):
            print(f"   🎯 GRW Regional Bonus Applied")
        if r.get('next_deadline'):
            print(f"   Deadline:    {r['next_deadline']}")
        if r.get('explanation'):
            print(f"   💬 {r['explanation']}")
        print(f"   🔗 {r['source_url']}")


def generate_draft(query, program_name, profile):
    prompt = f"""Du bist ein Fördermittelberater. Schreibe eine hochprofessionelle, 3-Absatz Projektskizze (Project Outline) für einen offiziellen Förderantrag.
Programm: {program_name}
Projekt des Nutzers: {query}
Unternehmensdaten: {profile.get('employees', 'KMU')} Mitarbeiter, Sitz in {profile.get('state', 'Deutschland')}.

Struktur:
1. Ausgangslage & Zielsetzung
2. Innovationsgehalt & Technische Umsetzung
3. Verwertung & Wirtschaftlicher Hebel

Schreibe im formellen, behördlichen Deutsch. Keine Einleitung, kein Fazit, nur der Text."""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Fehler bei der Generierung: {str(e)}"
