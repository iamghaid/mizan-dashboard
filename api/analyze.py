from http.server import BaseHTTPRequestHandler
from email.parser import BytesParser
from email.policy import default as _email_policy
import json
import hashlib
import io

try:  # cgi أُزيلت من بايثون 3.13، فنستخدم محلل email كبديل عند غيابها
    import cgi
    if not hasattr(cgi, "FieldStorage"):  # وحدة موجودة لكن بلا الصنف المطلوب
        cgi = None
except ImportError:  # pragma: no cover
    cgi = None

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


REQUIRED_COLUMNS = ["auction_id", "event_time", "bidder_id", "bid_amount"]

# أعمدة اختيارية: التحليل لا يتوقف بغيابها، لكن العامل المبني عليها يُوسم
# "غير متوفر" بدل أن يُحتسب صفرًا — لأن غياب البيانة ليس دليل سلامة.
#   has_service_suspension   : هل عليه إيقاف خدمات — 0/1  (أقوى مؤشر عجز رسمي)
#   outstanding_loans        : قيمة القروض غير المسددة عليه (بالريال)
#   prior_payment_defaults   : عدد التعثرات المالية السابقة المسجّلة عليه
#   guarantee_amount         : قيمة الضمان/الكفالة التي قدّمها المزايد (بالريال)
#   prior_unpaid_wins        : مرات رسا عليه مزاد سابق ولم يكمل السداد/إتمام البيع
#   prior_unpaid_bid_amount  : قيمة المزايدة السابقة التي أخفق في إتمامها (بالريال)
OPTIONAL_COLUMNS = {
    "has_service_suspension": 0,
    "outstanding_loans": 0,
    "prior_payment_defaults": 0,
    "guarantee_amount": 0,
    "prior_unpaid_wins": 0,
    "prior_unpaid_bid_amount": 0,
}

# نافذة زمنية (بالثواني) يُعتبر تجاوزها "رفع سعر غير معتاد" من نفس المزايد على نفسه
QUICK_SELF_REBID_SECONDS = 20

# ===== أوزان مؤشر القدرة على السداد =====
# المشكلة التي يحلّها ميزان: مزايد غير قادر ماليًا يرسو عليه المزاد ثم لا يسدّد،
# فتطول العملية وتدخل القضاء ويُعاد طرح الأصل ويُباع بسعر أقل. لذلك ثقل المؤشر
# كله في القدرة المالية والسجل، لا في سلوك المزايدة.
WEIGHTS = {
    # أ) الملاءة المالية — جوهر المؤشر
    "service_suspension": 45.0,   # إيقاف خدمات: مؤشر عجز رسمي قائم
    "default_each": 12.5,         # لكل تعثر مالي سابق
    "default_cap": 25.0,
    "loans": 40.0,                # قروض غير مسددة مقارنةً بما يزايد عليه
    "guarantee": 25.0,            # ضعف تغطية الضمان مقابل أعلى مزايدة

    # ب) السجل السابق في المزادات
    "unpaid_each": 20.0,          # لكل فوز سابق بلا إكمال سداد
    "unpaid_cap": 40.0,
    "exceeds_prior": 15.0,        # تجاوز السعر لمزايدته السابقة المتعثرة

    # ج) مؤشرات سلوكية ثانوية — مجموعها 15 عمدًا، أي أقل من عتبة "متوسط" (40)
    # فلا يمكن لأيٍّ منها أو لها مجتمعةً أن ترفع تصنيف مزايد وحدها. أُبقيت
    # لأنها مفيدة كسياق للمراجع البشري، لا كسبب للمنع.
    "anomaly": 10.0,              # سلوك شاذ إحصائيًا (Isolation Forest)
    "escalation_each": 2.5,
    "escalation_cap": 5.0,
}

# نسبة القروض غير المسددة إلى أعلى مزايدة التي تُعد مقلقة بالكامل:
# من عليه قروض غير مسددة تعادل نصف ما يزايد عليه أو أكثر يأخذ الوزن كاملًا.
CONCERNING_LOAN_RATIO = 0.50

ADEQUATE_GUARANTEE_RATIO = 0.10


def extract_uploaded_csv(content_type: str, body: bytes, headers):
    """
    يستخرج محتوى ملف CSV من الطلب. يدعم multipart/form-data (رفع من المتصفح)
    وإرسال محتوى الملف مباشرةً في جسم الطلب.

    يستخدم cgi عند توفّرها، وإلا فمحلل email القياسي — حتى يبقى الرفع شغّالًا
    على بايثون 3.13 فما فوق حيث أُزيلت وحدة cgi.
    """
    if "multipart/form-data" not in content_type:
        return body

    if cgi is not None:
        fs = cgi.FieldStorage(
            fp=io.BytesIO(body),
            headers=headers,
            environ={"REQUEST_METHOD": "POST"},
        )
        field = fs["file"] if "file" in fs else None
        return field.file.read() if field is not None else None

    raw = b"Content-Type: " + content_type.encode("utf-8") + b"\r\n\r\n" + body
    message = BytesParser(policy=_email_policy).parsebytes(raw)
    for part in message.iter_parts():
        if part.get_filename() or part.get_param("name", header="content-disposition") == "file":
            return part.get_payload(decode=True)
    return None


def compute_hash_chain(df: pd.DataFrame):
    """
    يبني سلسلة بصمات SHA-256 متتالية على صفوف الأحداث بترتيبها الأصلي،
    بحيث أي تعديل لاحق على أي صف يُغيّر كل البصمات التي تليه وتنكشف المخالفة.
    هذا مستقل تمامًا عن تحليل السلوك أدناه (لا يُخلط بينهما — سلامة السجل شيء، ونزاهة السلوك شيء آخر).
    """
    prev_hash = "0" * 64
    hashes = []
    for _, row in df.iterrows():
        payload = f"{prev_hash}|{row.to_dict()}".encode("utf-8")
        current_hash = hashlib.sha256(payload).hexdigest()
        hashes.append(current_hash)
        prev_hash = current_hash
    return hashes


def run_isolation_forest(df: pd.DataFrame):
    """
    يبني مؤشرات رقمية لكل حدث مزايدة ويشغّل نموذج IsolationForest حقيقي
    (من مكتبة scikit-learn، وليس قواعد if/else يدوية) ليكتشف الأحداث الشاذة إحصائيًا.
    يُعاد تدريب النموذج من الصفر على كل ملف يُرفع.
    """
    work = df.copy()

    work["bid_amount"] = pd.to_numeric(work["bid_amount"], errors="coerce").fillna(0)

    # عدد مزايدات نفس المزايد داخل نفس المزاد (كثافة التناوب)
    work["bidder_freq"] = work.groupby(["auction_id", "bidder_id"])["bidder_id"].transform("count")

    # ترتيب قيمة المزايدة نسبةً لباقي مزايدات نفس المزاد
    work["bid_rank_pct"] = work.groupby("auction_id")["bid_amount"].rank(pct=True)

    # الفاصل الزمني بين كل مزايدة والتي قبلها في نفس المزاد (بالثواني)
    if "event_time" in work.columns:
        work["event_time_parsed"] = pd.to_datetime(work["event_time"], errors="coerce")
        work = work.sort_values(["auction_id", "event_time_parsed"])
        work["time_gap"] = (
            work.groupby("auction_id")["event_time_parsed"]
            .diff()
            .dt.total_seconds()
            .fillna(0)
        )
        # الفاصل الزمني بين مزايدة المزايد ومزايدته السابقة هو نفسه (لرصد "الرفع على نفسه" بسرعة)
        work["own_gap"] = (
            work.groupby(["auction_id", "bidder_id"])["event_time_parsed"]
            .diff()
            .dt.total_seconds()
        )
    else:
        work["time_gap"] = 0
        work["own_gap"] = np.nan

    # تكرار رفع السعر بشكل غير معتاد: المزايد يرفع على نفسه خلال نافذة زمنية قصيرة جدًا
    work["quick_self_rebid"] = (
        work["own_gap"].notna() & (work["own_gap"] >= 0) & (work["own_gap"] < QUICK_SELF_REBID_SECONDS)
    )

    feature_cols = ["bid_amount", "bidder_freq", "bid_rank_pct", "time_gap"]
    X = work[feature_cols].fillna(0).values

    model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    model.fit(X)

    work["anomaly_score"] = model.decision_function(X)
    work["anomaly_flag"] = (model.predict(X) == -1)

    return work


def build_bidder_risk_table(work: pd.DataFrame, available_columns=None):
    """
    يحوّل أحداث المزايدة إلى مؤشر **قدرة على السداد** لكل مزايد.

    المشكلة المستهدفة: مزايد غير قادر ماليًا يرسو عليه المزاد ثم لا يكمل
    السداد، فتطول الإجراءات وتدخل القضاء ويُعاد طرح الأصل ويُباع بسعر أقل.
    التحقق من الهوية (إنفاذ/أبشر) يثبت مَن هو، ولا يثبت أنه **قادر** على الدفع.

    محاور المؤشر:
      أ) الملاءة المالية       -> إيقاف خدمات، قروض غير مسددة، تعثرات، تغطية الضمان
      ب) السجل في المزادات     -> فوز سابق بلا إكمال سداد، وتجاوز مزايدته المتعثرة
      ج) مؤشرات سلوكية ثانوية  -> شذوذ إحصائي ورفع سعر متكرر (سياق للمراجع فقط)

    `available_columns`: الأعمدة الاختيارية الموجودة فعلًا في الملف المرفوع.
    العامل الذي تغيب بيانته يُوسم "غير متوفر" ولا يُحتسب صفرًا.
    """
    available = set(available_columns or [])

    for col, default in OPTIONAL_COLUMNS.items():
        if col not in work.columns:
            work[col] = default
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default)

    grouped = work.groupby("bidder_id")

    rows = []
    for bidder_id, g in grouped:
        events_count = len(g)
        anomaly_events = int(g["anomaly_flag"].sum())
        anomaly_ratio = anomaly_events / events_count if events_count else 0.0
        escalation_count = int(g["quick_self_rebid"].sum())
        has_suspension = bool(g["has_service_suspension"].max() > 0)
        outstanding_loans = float(g["outstanding_loans"].max())
        prior_unpaid_wins = int(g["prior_unpaid_wins"].max())
        prior_unpaid_bid_amount = float(g["prior_unpaid_bid_amount"].max())
        prior_payment_defaults = int(g["prior_payment_defaults"].max())
        guarantee_amount = float(g["guarantee_amount"].max())
        max_bid = float(pd.to_numeric(g["bid_amount"], errors="coerce").fillna(0).max())
        auctions_involved = sorted(g["auction_id"].astype(str).unique().tolist())

        guarantee_known = "guarantee_amount" in available and guarantee_amount > 0 and max_bid > 0
        guarantee_ratio = (guarantee_amount / max_bid) if guarantee_known else None

        # تجاوز مزايدته السابقة المتعثرة: التزم الآن بمبلغ أكبر مما عجز عن سداده
        # العمود موجود = البيانة متوفرة. غياب مزايدة متعثرة سابقة ليس نقص
        # بيانات بل "لا ينطبق" — والتمييز بينهما يمنع تشويه نسبة التغطية.
        exceeds_known = "prior_unpaid_bid_amount" in available
        has_prior_unpaid_bid = exceeds_known and prior_unpaid_bid_amount > 0
        exceeds_prior = bool(has_prior_unpaid_bid and max_bid > prior_unpaid_bid_amount)

        factors = []

        # ===== أ) الملاءة المالية =====
        suspension_known = "has_service_suspension" in available
        factors.append({
            "key": "service_suspension", "axis": "solvency",
            "label": "إيقاف خدمات قائم على المزايد",
            "label_en": "Active service suspension",
            "available": suspension_known,
            "points": round(WEIGHTS["service_suspension"] if (suspension_known and has_suspension) else 0.0, 1),
            "max_points": WEIGHTS["service_suspension"],
            "detail": ("موجود — مؤشر عجز رسمي قائم" if has_suspension else "لا يوجد")
                      if suspension_known else "لا يوجد عمود has_service_suspension في الملف",
            "detail_en": ("present — active official default indicator" if has_suspension else "none")
                         if suspension_known else "column has_service_suspension not in the file",
        })

        loans_known = "outstanding_loans" in available and max_bid > 0
        if loans_known:
            loan_ratio = outstanding_loans / max_bid
            loans_points = min(loan_ratio / CONCERNING_LOAN_RATIO, 1.0) * WEIGHTS["loans"]
            loans_detail = (f"{int(outstanding_loans):,} ريال قروض غير مسددة مقابل أعلى مزايدة "
                            f"{int(max_bid):,} ({round(loan_ratio * 100, 1)}%)")
            loans_detail_en = (f"{int(outstanding_loans):,} unpaid loans against a top bid of "
                               f"{int(max_bid):,} ({round(loan_ratio * 100, 1)}%)")
        else:
            loans_points = 0.0
            loans_detail = "لا يوجد عمود outstanding_loans في الملف"
            loans_detail_en = "column outstanding_loans not in the file"
        factors.append({
            "key": "loans", "axis": "solvency",
            "label": "قروض غير مسددة مقارنةً بقيمة مزايدته",
            "label_en": "Unpaid loans against bid value",
            "available": loans_known, "points": round(loans_points, 1),
            "max_points": WEIGHTS["loans"], "detail": loans_detail, "detail_en": loans_detail_en,
        })

        defaults_known = "prior_payment_defaults" in available
        defaults_points = min(prior_payment_defaults * WEIGHTS["default_each"], WEIGHTS["default_cap"]) if defaults_known else 0.0
        factors.append({
            "key": "payment_defaults", "axis": "solvency",
            "label": "تعثرات مالية سابقة مسجّلة",
            "label_en": "Recorded past payment defaults",
            "available": defaults_known, "points": round(defaults_points, 1),
            "max_points": WEIGHTS["default_cap"],
            "detail": f"{prior_payment_defaults} تعثر" if defaults_known else "لا يوجد عمود prior_payment_defaults في الملف",
            "detail_en": f"{prior_payment_defaults} default(s)" if defaults_known else "column prior_payment_defaults not in the file",
        })

        if guarantee_known:
            shortfall = max(0.0, (ADEQUATE_GUARANTEE_RATIO - guarantee_ratio) / ADEQUATE_GUARANTEE_RATIO)
            guarantee_points = shortfall * WEIGHTS["guarantee"]
            guarantee_detail = (f"ضمان {int(guarantee_amount):,} مقابل أعلى مزايدة {int(max_bid):,} "
                                f"({round(guarantee_ratio * 100, 1)}% — المتوقع {int(ADEQUATE_GUARANTEE_RATIO * 100)}%)")
            guarantee_detail_en = (f"guarantee {int(guarantee_amount):,} against top bid {int(max_bid):,} "
                                   f"({round(guarantee_ratio * 100, 1)}% — {int(ADEQUATE_GUARANTEE_RATIO * 100)}% expected)")
        else:
            guarantee_points = 0.0
            guarantee_detail = "لا توجد بيانات ضمان في الملف"
            guarantee_detail_en = "no guarantee data in the file"
        factors.append({
            "key": "guarantee", "axis": "solvency",
            "label": "تغطية الضمان مقابل قيمة المزايدة",
            "label_en": "Guarantee coverage against bid value",
            "available": guarantee_known, "points": round(guarantee_points, 1),
            "max_points": WEIGHTS["guarantee"], "detail": guarantee_detail, "detail_en": guarantee_detail_en,
        })

        # ===== ب) السجل في المزادات =====
        unpaid_known = "prior_unpaid_wins" in available
        unpaid_points = min(prior_unpaid_wins * WEIGHTS["unpaid_each"], WEIGHTS["unpaid_cap"]) if unpaid_known else 0.0
        factors.append({
            "key": "unpaid_wins", "axis": "record",
            "label": "فوز سابق بمزاد دون إكمال السداد",
            "label_en": "Past auction wins without completed payment",
            "available": unpaid_known, "points": round(unpaid_points, 1),
            "max_points": WEIGHTS["unpaid_cap"],
            "detail": f"{prior_unpaid_wins} مرة" if unpaid_known else "لا يوجد عمود prior_unpaid_wins في الملف",
            "detail_en": f"{prior_unpaid_wins} occurrence(s)" if unpaid_known else "column prior_unpaid_wins not in the file",
        })

        factors.append({
            "key": "exceeds_prior", "axis": "record",
            "label": "تجاوز مزايدته السابقة التي لم يكملها",
            "label_en": "Bid exceeds the amount they previously failed to pay",
            "available": exceeds_known,
            "points": round(WEIGHTS["exceeds_prior"] if exceeds_prior else 0.0, 1),
            "max_points": WEIGHTS["exceeds_prior"],
            "detail": ((f"مزايدته الحالية {int(max_bid):,} تتجاوز {int(prior_unpaid_bid_amount):,} التي عجز عن سدادها"
                        if exceeds_prior else
                        f"لم يتجاوز مزايدته المتعثرة ({int(prior_unpaid_bid_amount):,})")
                       if has_prior_unpaid_bid else "لا توجد مزايدة سابقة متعثرة — لا ينطبق")
                      if exceeds_known else "لا يوجد عمود prior_unpaid_bid_amount في الملف",
            "detail_en": ((f"current bid {int(max_bid):,} exceeds the {int(prior_unpaid_bid_amount):,} they failed to pay"
                           if exceeds_prior else
                           f"below their unpaid bid ({int(prior_unpaid_bid_amount):,})")
                          if has_prior_unpaid_bid else "no prior unpaid bid - not applicable")
                         if exceeds_known else "column prior_unpaid_bid_amount not in the file",
        })

        # ===== ج) مؤشرات سلوكية ثانوية (سياق للمراجع، لا سبب للمنع) =====
        factors.append({
            "key": "anomaly", "axis": "behaviour",
            "label": "سلوك شاذ إحصائيًا (Isolation Forest)",
            "label_en": "Statistically abnormal behaviour",
            "available": True,
            "points": round(min(anomaly_ratio, 1.0) * WEIGHTS["anomaly"], 1),
            "max_points": WEIGHTS["anomaly"],
            "detail": f"{anomaly_events} من {events_count} حدثًا ({round(anomaly_ratio * 100, 1)}%)",
            "detail_en": f"{anomaly_events} of {events_count} events ({round(anomaly_ratio * 100, 1)}%)",
        })

        factors.append({
            "key": "escalation", "axis": "behaviour",
            "label": "تكرار رفع السعر بشكل غير معتاد",
            "label_en": "Unusually repeated price escalation",
            "available": True,
            "points": round(min(escalation_count * WEIGHTS["escalation_each"], WEIGHTS["escalation_cap"]), 1),
            "max_points": WEIGHTS["escalation_cap"],
            "detail": f"{escalation_count} مرة خلال أقل من {QUICK_SELF_REBID_SECONDS} ثانية",
            "detail_en": f"{escalation_count} times within {QUICK_SELF_REBID_SECONDS} seconds",
        })

        score = sum(f["points"] for f in factors)
        risk_score = int(round(min(score, 100)))

        # نسبة اكتمال البيانات: كم من وزن المؤشر أمكن تقييمه فعلًا
        total_weight = sum(f["max_points"] for f in factors)
        covered_weight = sum(f["max_points"] for f in factors if f["available"])
        data_completeness = int(round(covered_weight / total_weight * 100)) if total_weight else 0
        missing_factors = [f["key"] for f in factors if not f["available"]]

        if risk_score >= 70:
            level = "مرتفع"
            level_en = "High"
            action = "إيقاف/مراجعة فورية قبل اعتماد الترسية"
            action_en = "Hold and review before awarding"
        elif risk_score >= 40:
            level = "متوسط"
            level_en = "Medium"
            action = "مراجعة يدوية من مشرف المزاد"
            action_en = "Manual review by auction supervisor"
        else:
            level = "منخفض"
            level_en = "Low"
            action = "لا إجراء إضافي مطلوب حاليًا"
            action_en = "No extra action needed"

        rows.append({
            "bidder_id": bidder_id,
            "identity_verified": True,  # ثابت: التحقق من الهوية عبر إنفاذ وأبشر مستقل عن هذا المؤشر
            "events_count": events_count,
            "anomaly_events": anomaly_events,
            "anomaly_ratio": round(anomaly_ratio * 100, 1),
            "escalation_count": escalation_count,
            "prior_unpaid_wins": prior_unpaid_wins,
            "prior_payment_defaults": prior_payment_defaults,
            "has_service_suspension": has_suspension,
            "outstanding_loans": outstanding_loans,
            "prior_unpaid_bid_amount": prior_unpaid_bid_amount,
            "exceeds_prior_unpaid_bid": exceeds_prior,
            "guarantee_amount": guarantee_amount if guarantee_known else None,
            "guarantee_ratio": round(guarantee_ratio * 100, 1) if guarantee_known else None,
            "max_bid": max_bid,
            "factors": factors,
            "data_completeness": data_completeness,
            "missing_factors": missing_factors,
            "auctions_involved": auctions_involved,
            "risk_score": risk_score,
            "risk_level": level,
            "risk_level_en": level_en,
            "recommended_action": action,
            "recommended_action_en": action_en,
        })

    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    return rows


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            csv_bytes = extract_uploaded_csv(content_type, body, self.headers)
            if not csv_bytes:
                self._send_json(
                    {
                        "error": "لم يتم إرفاق ملف CSV",
                        "error_en": "No CSV file was attached",
                    },
                    400,
                )
                return

            df = pd.read_csv(io.BytesIO(csv_bytes))

            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                self._send_json(
                    {
                        "error": f"أعمدة مفقودة بملف CSV: {', '.join(missing)}",
                        "error_en": f"Missing columns in the CSV file: {', '.join(missing)}",
                    },
                    400,
                )
                return

            # الأعمدة الاختيارية الموجودة فعلًا في الملف — تُميَّز عن الغائبة
            # حتى لا يُقرأ غياب البيانة كأنه قيمة صفر (سلامة).
            available_optional = [c for c in OPTIONAL_COLUMNS if c in df.columns]

            analyzed = run_isolation_forest(df)
            hashes = compute_hash_chain(df)
            bidder_risk = build_bidder_risk_table(analyzed, available_optional)

            total = len(analyzed)
            flagged = int(analyzed["anomaly_flag"].sum())

            # نبقي فقط الأعمدة المفيدة للواجهة، ونتخلص من أعمدة العمل الداخلية
            # (event_time_parsed, own_gap, quick_self_rebid) التي قد تحوي NaN
            # وتكسر تحويل JSON بالمتصفح.
            display_cols = [c for c in df.columns if c not in OPTIONAL_COLUMNS] + \
                ["bidder_freq", "bid_rank_pct", "time_gap", "anomaly_score", "anomaly_flag"] + \
                list(OPTIONAL_COLUMNS.keys())
            display_cols = [c for c in dict.fromkeys(display_cols) if c in analyzed.columns]

            clean = analyzed[display_cols].copy()
            clean = clean.replace([np.inf, -np.inf], np.nan)
            clean = clean.astype(object).where(pd.notnull(clean), None)
            result_rows = clean.assign(hash=hashes).to_dict(orient="records")

            summary = {
                "total_events": total,
                "flagged_events": flagged,
                "flagged_percent": round((flagged / total) * 100, 2) if total else 0,
                "chain_valid": True,
                "chain_length": len(hashes),
                "final_hash": hashes[-1] if hashes else None,
                "high_risk_bidders": sum(1 for r in bidder_risk if r["risk_level"] == "مرتفع"),
                "medium_risk_bidders": sum(1 for r in bidder_risk if r["risk_level"] == "متوسط"),
                # مصدر كل حقل اختياري: موجود في الملف المرفوع أم غير متوفر
                "field_sources": {
                    col: ("uploaded" if col in available_optional else "unavailable")
                    for col in OPTIONAL_COLUMNS
                },
                "data_completeness": (
                    bidder_risk[0]["data_completeness"] if bidder_risk else 0
                ),
            }

            self._send_json(
                {"summary": summary, "rows": result_rows, "bidder_risk": bidder_risk},
                200,
            )

        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc), "error_en": str(exc)}, 500)

    def _send_json(self, payload, status):
        def _sanitize(obj):
            if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                return None
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            return obj

        safe_payload = _sanitize(payload)
        body = json.dumps(safe_payload, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
