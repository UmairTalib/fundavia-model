import json
import os
from groq import Groq

# Fetch the API key securely from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY environment variable is not set! The LLM will not work.")

MODEL = "openai/gpt-oss-120b"

# ─────────────────────────────────────────────
# STRICT JSON SCHEMA (enforced at extraction)
# ─────────────────────────────────────────────
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "industry":        {"type": ["string", "null"]},
        "state":           {"type": ["string", "null"]},
        "employees":       {"type": ["string", "integer", "null"]},
        "company_age":     {"type": ["string", "null"]},
        "project_desc":    {"type": ["string", "null"]},
        "project_status":  {"type": ["string", "null"], "enum": ["not_started", "already_started", "unknown", None]},
        "inferred_project_type": {"type": ["string", "null"]},  # AI deduces, Python confirms
        "needs_confirmation":    {"type": ["array", "null"]},    # Fields that need user confirmation
        "trl_level":       {"type": ["integer", "null"]},
        "rd_cooperation":  {"type": ["boolean", "null"]},
        "de_minimis":      {"type": ["boolean", "null"]},
        "is_off_topic":    {"type": ["boolean", "null"]}
    }
}

# ─────────────────────────────────────────────
# EXTRACTION PROMPT  (Single-Pass, Derive & Confirm)
# ─────────────────────────────────────────────
EXTRACTION_PROMPT = """Du bist ein spezialisiertes Extraktions-System für den deutschen Fördermittelberater Fundavia.

Deine einzige Aufgabe: Extrahiere relevante Fakten aus der Nachricht des Users und gib sie als valides JSON zurück.

WICHTIGE REGELN:
1. Felder, die der User NICHT erwähnt hat: WEGLASSEN (nicht null, nicht raten).
2. Felder, die der User NICHT weiß (z.B. "weiß nicht"): als "unknown" markieren.
3. Du darfst Kategorien ABLEITEN, aber musst sie dann als "needs_confirmation" markieren.
4. Niemals Bundesland, Mitarbeiterzahl oder Projektstatus erfinden oder raten.

ABLEITUNGSREGELN (Derive & Confirm):
- Wenn User eine neue Software/App/KI/Algorithmus entwickelt → inferred_project_type: "Forschung & Entwicklung (FuE)"
- Wenn User Bestandssoftware kauft oder implementiert → inferred_project_type: "Standard-Digitalisierung"
- Wenn User Mitarbeiter schult oder weiterbildet → inferred_project_type: "Qualifizierung & Weiterbildung"
- Wenn User Solaranlage, Heizung, Dämmung plant → inferred_project_type: "Energieeffizienz & Klimaschutz"
- Wenn TU/Hochschule/Fraunhofer erwähnt → rd_cooperation: true (ohne Bestätigung, da faktisch klar)
- Wenn "schon gekauft", "bereits angefangen", "rückwirkend" → project_status: "already_started"

ERLAUBTE FELDER:
- "industry": Branche als Freitext (z.B. "Bäckerei", "Software", "Maschinenbau")
- "state": Bundesland (z.B. "Bayern", "Berlin")
- "employees": Zahl oder "unknown"
- "company_age": z.B. "< 5 Jahre", "5 - 10 Jahre", "> 10 Jahre", "unknown"
- "project_desc": Kurze Zusammenfassung des Vorhabens in eigenen Worten
- "project_status": "not_started" | "already_started" | "unknown"
- "inferred_project_type": Kategorie die du abgeleitet hast (muss in needs_confirmation!)
- "needs_confirmation": Liste der Felder, die der User bestätigen soll (z.B. ["inferred_project_type"])
- "trl_level": Integer 1-9
- "rd_cooperation": Boolean
- "de_minimis": Boolean
- "is_off_topic": true nur wenn komplett irrelevant (Witze, Rezepte, Code etc.)

Antworte NUR mit JSON. Keine Erklärungen."""

# ─────────────────────────────────────────────
# PLANNING PROMPT  (LLM decides what to ask next)
# ─────────────────────────────────────────────
PLANNING_PROMPT = """Du bist ein professioneller Förderberater bei Fundavia. Du führst ein natürliches Gespräch, um das Profil eines Unternehmens zu erfassen.

Dein aktuelles Wissen über den User:
{profile_summary}

Felder die noch fehlen (null): {missing_fields}
Felder die bestätigt werden müssen: {pending_confirmations}
Letzter User-Satz: "{last_message}"

AUFGABEN:
1. Falls "pending_confirmations" nicht leer ist: Bestätige deinen Vorschlag auf natürliche Weise.
   Beispiel: "Da Sie eine neue App entwickeln, würde ich das als Forschung & Entwicklung einstufen — stimmt das?"
2. Falls der User "weiß nicht" gesagt hat: Akzeptiere es freundlich und gehe weiter.
3. Fasse mehrere Fragen ZUSAMMEN wenn es natürlich wirkt (max. 2 auf einmal).
4. Beziehe dich auf das was der User gerade gesagt hat. Kein Roboter-Ton.
5. NIEMALS nach Branche oder Projektkategorie fragen — das leitest du ab.

PRIORITÄT der fehlenden Felder:
1. state (Bundesland) — wichtigste für Datenbankfilter
2. employees (Mitarbeiterzahl) — für KMU-Grenze
3. project_status (Gestartet?) — für Rückwirkend-Sperre
4. company_age (Unternehmensalter) — für Startup-Programme
5. rd_cooperation, trl_level — nur wenn FuE erkannt

Schreibe NUR die Antwort an den User. Kein JSON, keine Erklärungen."""

# ─────────────────────────────────────────────
# FIELDS REQUIRED BEFORE DATABASE QUERY
# ─────────────────────────────────────────────
REQUIRED_FIELDS = ["state", "employees", "project_status", "company_age"]
RD_EXTRA_FIELDS = ["trl_level", "rd_cooperation"]


class ConversationAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.history = []  # Full conversation memory for context
        self.profile = {
            "industry": None,
            "state": None,
            "employees": None,
            "company_age": None,
            "project_desc": None,
            "project_status": None,
            "inferred_project_type": None,
            "trl_level": None,
            "rd_cooperation": None,
            "de_minimis": None
        }
        self.pending_confirmations = []  # Fields derived but not yet user-confirmed
        self.flags = {
            "rd_detected": False,
            "retroactive_detected": False,
            "is_off_topic": False,
            "profile_complete": False
        }

    # ── STEP 1: Extract facts from user message ──────────────────────────
    def extract(self, user_msg: str) -> dict:
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user",   "content": user_msg}
        ]
        try:
            res = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
            raw = res.choices[0].message.content
            data = json.loads(raw)
            print(f"🧩 Extracted: {data}")
            return data
        except Exception as e:
            print(f"❌ Extraction Error: {e}")
            return {}

    # ── STEP 2: Update profile state ─────────────────────────────────────
    def update_profile(self, extracted: dict, user_msg: str):
        # Store conversation turn in memory
        self.history.append({"role": "user", "content": user_msg})

        # Handle off-topic flag
        self.flags["is_off_topic"] = bool(extracted.get("is_off_topic", False))
        if self.flags["is_off_topic"]:
            return

        # Handle pending confirmations for derived fields
        needs_conf = extracted.get("needs_confirmation", []) or []
        for field in needs_conf:
            if field not in self.pending_confirmations:
                self.pending_confirmations.append(field)

        # Update profile with extracted values
        for k, v in extracted.items():
            if k in self.profile and v is not None:
                # "unknown" is a valid non-blocking answer
                self.profile[k] = v

        # Python Legal Firewalls
        # 1. R&D trigger
        desc = str(self.profile.get("project_desc") or "").lower()
        inferred = str(self.profile.get("inferred_project_type") or "").lower()
        if any(w in desc for w in ["forschung", "entwicklung", "ki ", "software", "algorithmus", "neue app", "neue plattform"]) \
           or "fue" in inferred or "forschung" in inferred:
            self.flags["rd_detected"] = True

        # 2. Retroactive firewall
        if self.profile.get("project_status") == "already_started":
            self.flags["retroactive_detected"] = True
            print("🛡️ Retroactive Firewall Triggered!")

        # 3. Check if profile is complete
        self._check_completeness()

    # ── STEP 3: Deterministic completeness check (Python legal gate) ─────
    def _check_completeness(self):
        required = list(REQUIRED_FIELDS)
        if self.flags["rd_detected"]:
            required += RD_EXTRA_FIELDS

        all_filled = all(
            self.profile.get(f) is not None
            for f in required
        )
        no_pending = len(self.pending_confirmations) == 0

        self.flags["profile_complete"] = all_filled and no_pending
        if self.flags["profile_complete"]:
            print("✅ Profile complete — ready to query database.")

    # ── STEP 4: Generate the next natural conversational reply ────────────
    def plan_next_reply(self, user_msg: str) -> str:
        # Guardrail: off-topic deflection (hardcoded, no LLM)
        if self.flags["is_off_topic"]:
            print("🛡️ Guardrail: Off-topic deflected.")
            missing = self._get_missing_fields()
            next_topic = missing[0] if missing else "Ihr Vorhaben"
            return (
                "Ich bin ein spezialisierter KI-Förderberater und kann leider nicht zu "
                f"anderen Themen Auskunft geben. Lassen Sie uns weitermachen: "
                f"Könnten Sie mir noch kurz sagen, {self._field_label(next_topic)}?"
            )

        # Guardrail: retroactive — immediate flag, no more questions
        if self.flags["retroactive_detected"] and self.flags["profile_complete"]:
            return None  # Signal to fire engine immediately

        # Build a profile summary for the planning LLM
        profile_summary = self._summarize_profile()
        missing = self._get_missing_fields()
        pending = self.pending_confirmations

        if not missing and not pending:
            return None  # Profile complete, signal to fire engine

        prompt = PLANNING_PROMPT.format(
            profile_summary=profile_summary,
            missing_fields=", ".join(missing) if missing else "keine",
            pending_confirmations=", ".join(pending) if pending else "keine",
            last_message=user_msg
        )

        try:
            res = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.4
            )
            reply = res.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"❌ Planning Error: {e}")
            return "Könnten Sie mir noch etwas mehr über Ihr Unternehmen erzählen?"

    # ── Handle confirmation response from user ───────────────────────────
    def handle_confirmation(self, user_msg: str, field: str):
        """If user confirmed a derived field, accept it. If denied, clear it."""
        negative = any(w in user_msg.lower() for w in ["nein", "falsch", "nicht", "no ", "stimmt nicht"])
        if negative:
            self.profile[field] = None
            print(f"🔄 User rejected inference for: {field}")
        if field in self.pending_confirmations:
            self.pending_confirmations.remove(field)
        self._check_completeness()

    # ── Build the form_data dict for the SQL engine ───────────────────────
    def get_form_data(self) -> dict:
        return {
            "employees":     self.profile.get("employees"),
            "state":         self.profile.get("state"),
            "district":      self.profile.get("district"),
            "company_age":   self.profile.get("company_age"),
            "project_type":  self.profile.get("inferred_project_type"),
            "project_status": self.profile.get("project_status")
        }

    def get_query_text(self) -> str:
        return self.profile.get("project_desc") or self.profile.get("industry") or "Allgemein"

    # ── Helpers ───────────────────────────────────────────────────────────
    def _get_missing_fields(self) -> list:
        required = list(REQUIRED_FIELDS)
        if self.flags["rd_detected"]:
            required += RD_EXTRA_FIELDS
        return [f for f in required if self.profile.get(f) is None]

    def _summarize_profile(self) -> str:
        parts = []
        if self.profile.get("industry"):        parts.append(f"Branche: {self.profile['industry']}")
        if self.profile.get("state"):           parts.append(f"Bundesland: {self.profile['state']}")
        if self.profile.get("employees"):       parts.append(f"Mitarbeiter: {self.profile['employees']}")
        if self.profile.get("company_age"):     parts.append(f"Alter: {self.profile['company_age']}")
        if self.profile.get("project_desc"):    parts.append(f"Vorhaben: {self.profile['project_desc']}")
        if self.profile.get("project_status"):  parts.append(f"Status: {self.profile['project_status']}")
        if self.profile.get("inferred_project_type"): parts.append(f"Typ (abgeleitet): {self.profile['inferred_project_type']}")
        return ", ".join(parts) if parts else "Noch keine Informationen"

    def _field_label(self, field: str) -> str:
        labels = {
            "state": "in welchem Bundesland Ihr Unternehmen registriert ist",
            "employees": "wie viele Mitarbeiter Sie haben",
            "project_status": "ob das Projekt bereits gestartet ist",
            "company_age": "wann Ihr Unternehmen gegründet wurde",
            "trl_level": "auf welchem TRL-Niveau (1–9) sich Ihre Lösung befindet",
            "rd_cooperation": "ob Sie mit einer Hochschule oder Forschungseinrichtung zusammenarbeiten"
        }
        return labels.get(field, field)
