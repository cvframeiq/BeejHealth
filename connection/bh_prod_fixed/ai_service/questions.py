"""
BeejHealth — 30 Coconut Adaptive Questions
User ko sirf 5 dikhenge: Q1 (branch selector) + 4 branch questions
"""

COCONUT_Q1 = {
    "id": "cq1", "question": "Aapke naariyal ke ped mein sabse pehle kya symptom dikha?",
    "type": "single", "branch_selector": True,
    "options": [
        {"id": "spots",  "label": "🟤 Patti pe daag ya dhabba",         "desc": "Brown, gray ya kala daag"},
        {"id": "wilt",   "label": "🔴 Tana ya kali mein problem",        "desc": "Trunk se liquid ya kali gal rahi"},
        {"id": "yellow", "label": "🟡 Patti peeli ho rahi hai",          "desc": "Leaves yellow ya dry"},
        {"id": "none",   "label": "✅ Koi symptom nahi, preventive check","desc": "Tree healthy lagta hai"},
    ],
    "branch_map": {"spots": "coconut_spots", "wilt": "coconut_wilt", "yellow": "coconut_yellow", "none": "coconut_none"},
}

BRANCH_QUESTIONS = {

    # ── Branch 1: Patti pe daag → Gray Leaf Spot / Leaf Rot ──────────
    "coconut_spots": [
        {"id": "cq2", "question": "Daag ka rang kaisa hai?",
         "options": [{"id": "gray", "label": "⬜ Gray/Ash rang"}, {"id": "brown", "label": "🟫 Brown/Dark Brown"},
                     {"id": "yellow_edge", "label": "🟡 Brown with yellow border"}, {"id": "black", "label": "⬛ Kala daag"}]},
        {"id": "cq3", "question": "Daag patti ke kahan par hai?",
         "options": [{"id": "tip", "label": "🔝 Patti ki nauk pe"}, {"id": "middle", "label": "↔️ Beech mein"},
                     {"id": "base", "label": "⬇️ Neeche jaad ke paas"}, {"id": "all", "label": "📍 Poori patti pe"}]},
        {"id": "cq4", "question": "Kitni pattiyon pe daag hai?",
         "options": [{"id": "one", "label": "1️⃣ Ek ya do patti"}, {"id": "few", "label": "🔢 3-5 pattiyan"},
                     {"id": "many", "label": "📊 Adhi se zyada"}, {"id": "all", "label": "⚠️ Saari pattiyan"}]},
        {"id": "cq5", "question": "Daag ka size kaisa hai?",
         "options": [{"id": "small", "label": "🔵 Chota (1-2mm)"}, {"id": "medium", "label": "🟡 Medium (5-10mm)"},
                     {"id": "large", "label": "🔴 Bada (>1cm)"}, {"id": "merge", "label": "💀 Daag aapas mein jud gaye"}]},
    ],

    # ── Branch 2: Tana/kali problem → Stem Bleeding / Bud Rot ────────
    "coconut_wilt": [
        {"id": "cq7", "question": "Problem mainly kahan hai?",
         "options": [{"id": "trunk", "label": "🌴 Tane (trunk) pe"}, {"id": "bud", "label": "🌱 Nai kali/growth pe"},
                     {"id": "roots", "label": "🌿 Neeche jaad ke paas"}, {"id": "both", "label": "⚠️ Trunk aur kali dono"}]},
        {"id": "cq8", "question": "Tane se koi liquid nikal raha hai?",
         "options": [{"id": "dark_liquid", "label": "🩸 Haan, dark brown/red liquid"}, {"id": "clear", "label": "💧 Haan, transparent liquid"},
                     {"id": "no", "label": "❌ Nahi, koi liquid nahi"}, {"id": "smell", "label": "👃 Liquid hai + buri smell"}]},
        {"id": "cq9", "question": "Nai patti ya kali ka kya haal hai?",
         "options": [{"id": "normal", "label": "✅ Normal badi ho rahi hai"}, {"id": "brown", "label": "🟫 Kali brown/black ho gayi"},
                     {"id": "small", "label": "📉 Patti choti reh gayi"}, {"id": "dead", "label": "💀 Kali/patti mar gayi"}]},
        {"id": "cq10", "question": "Yeh problem kitne samay se hai?",
         "options": [{"id": "week", "label": "📅 1 hafte se kam"}, {"id": "month", "label": "🗓️ 1-2 mahine"},
                     {"id": "old", "label": "📆 3-6 mahine"}, {"id": "very_old", "label": "⏳ 6 mahine se zyada"}]},
    ],

    # ── Branch 3: Yellowing → Nutrient deficiency ─────────────────────
    "coconut_yellow": [
        {"id": "cq12", "question": "Yellowing kahan se shuru hua?",
         "options": [{"id": "old_leaves", "label": "⬇️ Neeche ki purani pattiyon se"}, {"id": "new_leaves", "label": "⬆️ Upar ki nai pattiyon se"},
                     {"id": "all_at_once", "label": "📊 Saari pattiyan ek saath"}, {"id": "patches", "label": "🗺️ Kuch jagah kuch jagah"}]},
        {"id": "cq13", "question": "Patti ki nasubon (veins) ka rang kaisa hai?",
         "options": [{"id": "green_vein", "label": "🟢 Nasuben hari, baaki peeli"}, {"id": "all_yellow", "label": "🟡 Sab kuch peela"},
                     {"id": "brown_vein", "label": "🟫 Nasuben bhi brown"}, {"id": "normal", "label": "✅ Nasuben theek hain"}]},
        {"id": "cq14", "question": "Paani dene ka schedule kya hai?",
         "options": [{"id": "too_much", "label": "💧 Bahut zyada paani"}, {"id": "less", "label": "🏜️ Kam paani milta hai"},
                     {"id": "regular", "label": "✅ Regular schedule pe"}, {"id": "rain", "label": "🌧️ Sirf baarish pe depend"}]},
        {"id": "cq15", "question": "Khaad (fertilizer) kab diya tha?",
         "options": [{"id": "recent", "label": "📅 1 mahine ke andar"}, {"id": "months", "label": "🗓️ 2-6 mahine pehle"},
                     {"id": "long", "label": "📆 6+ mahine nahi diya"}, {"id": "never", "label": "❌ Kabhi nahi diya"}]},
    ],

    # ── Branch 4: No symptoms → Preventive check ──────────────────────
    "coconut_none": [
        {"id": "cq17", "question": "Aap kaun sa coconut variety ughate hain?",
         "options": [{"id": "tall", "label": "🌴 Tall variety (Tiptur, West Coast)"}, {"id": "dwarf", "label": "🌿 Dwarf variety"},
                     {"id": "hybrid", "label": "🔬 Hybrid (Kera Sankara)"}, {"id": "unknown", "label": "❓ Pata nahi"}]},
        {"id": "cq18", "question": "Aapka farm kaisi jagah pe hai?",
         "options": [{"id": "coastal", "label": "🌊 Coastal (samundar ke paas)"}, {"id": "inland", "label": "🏔️ Inland (andar ki taraf)"},
                     {"id": "humid", "label": "💧 Bahut nami wali jagah"}, {"id": "dry", "label": "☀️ Sukha ilaqa"}]},
        {"id": "cq19", "question": "Last spray kab kiya tha?",
         "options": [{"id": "recent", "label": "📅 1 mahine ke andar"}, {"id": "months", "label": "🗓️ 2-4 mahine pehle"},
                     {"id": "long", "label": "📆 6+ mahine nahi kiya"}, {"id": "never", "label": "❌ Kabhi nahi"}]},
        {"id": "cq20", "question": "Aas paas ke pedo mein koi bimari dikhi?",
         "options": [{"id": "yes", "label": "⚠️ Haan, bahut pedo mein disease hai"}, {"id": "no", "label": "✅ Nahi, sab theek hain"},
                     {"id": "some", "label": "🔢 Kuch pedo mein thodi problem"}, {"id": "unknown", "label": "❓ Maine check nahi kiya"}]},
    ],
}

def get_session(q1_answer_id: str) -> dict:
    """Q1 answer ke baad 4 branch questions return karo"""
    branch_map = {"spots": "coconut_spots", "wilt": "coconut_wilt", "yellow": "coconut_yellow", "none": "coconut_none"}
    branch = branch_map.get(q1_answer_id, "coconut_none")
    branch_qs = BRANCH_QUESTIONS.get(branch, BRANCH_QUESTIONS["coconut_none"])
    return {
        "branch": branch,
        "total_questions": 5,
        "questions": [COCONUT_Q1] + branch_qs[:4],
    }
