"""
Massive Retraining — Sweep Neural Engine.

Generates 500+ high-quality training examples per domain,
trains with proper hyperparameters (15 epochs, warmup, scheduling),
and saves models that the NeuralEngine singleton loads automatically.

Usage:
    python -m sweep_neural_mesh.training.retrain_all
"""
from __future__ import annotations

import sys
import os
import json
import time
import logging
from pathlib import Path

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain")

OUTPUT_BASE = Path(__file__).parent / "neural_models"


# ════════════════════════════════════════════════════════════════
# MASSIVE TRAINING DATA — Evidence Classification
# ════════════════════════════════════════════════════════════════

EVIDENCE_DATA = []

# SUPPORTS (label 0) — 180+ examples
_supports = [
    "Studies confirm that regular exercise reduces cardiovascular disease risk by 30%.",
    "Meta-analysis of 50 trials shows the vaccine is 95% effective.",
    "Peer-reviewed research demonstrates that reading improves cognitive function.",
    "Longitudinal data shows the treatment group outperformed controls significantly.",
    "Multiple independent studies confirm the relationship between sleep and performance.",
    "Clinical trial results show statistically significant improvement in the treatment group.",
    "Expert consensus across 12 institutions supports this conclusion.",
    "The evidence from 3 independent research teams all points to the same result.",
    "Government data confirms the economic growth trend over the past 5 years.",
    "Systematic review of 200 studies confirms the safety profile.",
    "A peer-reviewed study in Nature demonstrated the mechanism clearly.",
    "Federal health agencies officially recommend this approach.",
    "The FDA approved the drug after rigorous testing phases.",
    "University researchers replicated the original findings across 3 labs.",
    "The WHO guidelines explicitly support this intervention.",
    "Randomized controlled trial showed 40% reduction in symptoms.",
    "Cohort study of 10,000 participants confirmed the association.",
    "Double-blind study demonstrated significant improvement over placebo.",
    "Network meta-analysis ranked this treatment as most effective.",
    "Real-world evidence from 50,000 patients supports efficacy.",
    "The intervention showed consistent benefits across all age groups.",
    "Biomarker analysis confirmed the biological mechanism.",
    "Cost-effectiveness analysis showed favorable outcomes.",
    "Long-term follow-up data confirmed sustained benefits.",
    "Multiple endpoints showed significant improvement.",
    "The research paper provides strong empirical support for the hypothesis.",
    "Experimental results validate the theoretical predictions made by the team.",
    "Post-hoc analysis confirmed the primary endpoint was met.",
    "The drug demonstrated superiority over existing standard of care.",
    "Three phase III trials all showed positive results.",
    "The systematic review found consistent evidence of benefit.",
    "Registry data from 200,000 patients confirmed the safety profile.",
    "The meta-analysis pooled data from 35 studies showing clear benefit.",
    "Real-world evidence supports the clinical trial findings.",
    "The intervention reduced hospitalization rates by 45%.",
    "Biomarkers improved significantly in the treatment group.",
    "Quality of life scores increased substantially with treatment.",
    "The drug received regulatory approval in 12 countries.",
    "International guidelines recommend this as first-line therapy.",
    "The evidence base has grown to over 100 supporting studies.",
    "All five primary endpoints were met with statistical significance.",
    "The treatment effect was consistent across all subgroups analyzed.",
    "Network meta-analysis showed this treatment ranked first.",
    "The drug was superior to placebo on all secondary endpoints.",
    "Long-term safety data from 5-year follow-up is reassuring.",
    "The benefit was observed regardless of baseline severity.",
    "This finding has been replicated in three independent cohorts.",
    "The treatment reduced mortality by 20% in the intent-to-treat population.",
    "Pre-specified subgroup analyses confirmed the benefit across demographics.",
    "The number needed to treat is favorable at 8 patients.",
    "Biomarker changes correlated with clinical improvement.",
    "The regulatory review found robust evidence of efficacy.",
    "No safety signals emerged from the post-marketing surveillance.",
    "The drug showed durable response rates over 24 months.",
    "The clinical benefit was confirmed by an independent data monitoring committee.",
    "The study design was rigorous with adequate statistical power.",
    "Results were consistent with prior Phase II findings.",
    "The drug's mechanism of action is well-characterized and supported.",
    "The trial met its primary endpoint at the pre-specified interim analysis.",
    "The treatment demonstrated superiority in both per-protocol and ITT populations.",
    "Patient-reported outcomes showed meaningful improvement.",
    "The drug was well-tolerated with a favorable safety profile.",
    "The NNT of 6 indicates strong clinical benefit.",
    "The effect size was large (Cohen's d = 0.8).",
    "The study was pre-registered and conducted with high methodological rigor.",
    "The benefits were clinically meaningful and statistically significant.",
    "The evidence strongly supports using this intervention in clinical practice.",
    "All major medical societies endorse this approach.",
    "The treatment has been shown to improve survival outcomes.",
    "The data consistently demonstrate benefit across multiple endpoints.",
    "The intervention is recommended by evidence-based clinical guidelines.",
    "The drug was effective even in patients with comorbidities.",
    "The trial showed a clear dose-response relationship.",
    "The benefits persisted after adjusting for confounders.",
    "The drug was superior to the active comparator.",
    "The study confirmed the expected pharmacodynamic effects.",
    "The clinical trial results are generalizable to real-world settings.",
    "The treatment showed rapid onset of action within 2 weeks.",
    "The study demonstrated both efficacy and safety simultaneously.",
    "The evidence shows a clear and consistent treatment effect.",
    "The drug was effective in both pediatric and adult populations.",
    "The intervention improved outcomes by 35% compared to control.",
    "The study used validated instruments to measure outcomes.",
    "The treatment effect was maintained throughout the study period.",
    "The drug was superior on all pre-specified endpoints.",
    "The meta-analysis included only high-quality randomized trials.",
    "The drug was effective regardless of genetic polymorphisms.",
    "The treatment improved both short-term and long-term outcomes.",
    "The study was adequately powered and well-designed.",
    "The drug showed a favorable benefit-to-risk ratio.",
    "The evidence strongly supports the use of this therapy.",
    "The trial results were consistent across all study centers.",
    "The drug demonstrated significant clinical benefit.",
    "The treatment reduced the risk of disease progression.",
    "The study confirmed the drug's efficacy in a diverse population.",
    "The intervention was associated with significantly fewer adverse events.",
    "The drug was effective in treatment-resistant cases.",
    "The trial showed a statistically significant improvement in primary endpoint.",
    "The benefit was clinically meaningful for patients.",
    "The drug was more effective than current standard therapy.",
    "The evidence supports early initiation of treatment.",
    "The study demonstrated a clear survival advantage.",
    "The drug was effective in both men and women equally.",
    "The treatment improved functional capacity significantly.",
    "The clinical trial met all its objectives.",
    "The drug has been validated in multiple clinical settings.",
    "The benefits of this approach are well-documented in the literature.",
    "The study provides Level I evidence supporting this treatment.",
    "The drug showed rapid and sustained improvement.",
    "The intervention reduced complications by 50%.",
    "The trial confirmed the hypothesis with high confidence.",
    "The drug was superior to all comparators tested.",
    "The benefits were observed across all age groups studied.",
    "The evidence firmly establishes the efficacy of this treatment.",
    "The treatment was effective in preventing disease recurrence.",
    "The study showed significant improvement in quality of life.",
    "The drug was effective when used as monotherapy.",
    "The clinical benefits were confirmed by independent reviewers.",
    "The treatment was effective in reducing symptom severity.",
    "The drug has a well-established efficacy profile.",
    "The evidence confirms the therapeutic value of this intervention.",
    "This treatment is supported by Level A evidence.",
    "The study clearly demonstrates the benefit of the intervention.",
    "The drug consistently outperformed placebo across all endpoints.",
    "The intervention showed robust efficacy in the pivotal trial.",
    "The clinical data strongly support the effectiveness of this treatment.",
    "The drug was effective in both acute and chronic settings.",
    "The treatment was superior to placebo with a large effect size.",
    "The study provides definitive evidence of clinical benefit.",
    "The drug demonstrated clear efficacy in the target population.",
    "The benefits of treatment significantly outweigh the risks.",
    "The clinical evidence firmly supports this therapeutic approach.",
    "The drug showed consistent efficacy across multiple studies.",
    "The treatment was effective and well-tolerated in clinical practice.",
    "The study confirms that this drug is highly effective.",
    "The evidence is overwhelming in favor of this treatment approach.",
    "The drug was significantly more effective than the control.",
    "The clinical trial validated the drug's therapeutic potential.",
    "The intervention improved patient outcomes substantially.",
    "The drug was effective in preventing disease progression.",
    "The evidence supports the broad use of this treatment.",
    "The study confirmed the drug's efficacy beyond doubt.",
    "The treatment showed remarkable efficacy in the trial.",
    "The drug was proven effective in rigorous clinical testing.",
    "The benefits were consistent and clinically significant.",
    "The drug demonstrated excellent efficacy and tolerability.",
    "The evidence clearly supports the use of this drug in clinical practice.",
    "The treatment achieved all primary and secondary endpoints.",
    "The drug was highly effective in the intention-to-treat analysis.",
    "The study showed dramatic improvement in patient outcomes.",
    "The drug was superior to standard care in every analysis.",
    "The treatment produced lasting clinical improvement.",
    "The drug's efficacy has been confirmed in multiple phase III trials.",
    "The intervention showed strong evidence of clinical benefit.",
    "The drug was effective in reducing both symptoms and complications.",
    "The clinical data unequivocally support this treatment.",
    "The study provides compelling evidence of efficacy.",
    "The drug was significantly better than placebo on all measures.",
    "The treatment improved clinical outcomes by a large margin.",
    "The drug demonstrated strong and consistent efficacy.",
    "The evidence conclusively demonstrates the drug's effectiveness.",
    "The clinical trial produced definitive results supporting this treatment.",
    "The drug was effective in real-world clinical settings.",
    "The treatment has been proven safe and effective in large trials.",
    "The drug showed impressive efficacy in the clinical trial.",
    "The evidence robustly supports the clinical use of this drug.",
]
for s in _supports:
    EVIDENCE_DATA.append((s, 0))

# REFUTES (label 1) — 180+ examples
_refutes = [
    "The drug showed no significant effect compared to placebo in the trial.",
    "Studies found no evidence supporting the claimed health benefits.",
    "The experiment failed to demonstrate any measurable improvement.",
    "Research contradicts the widely held belief about this topic.",
    "The data shows the intervention had zero measurable impact.",
    "Meta-analysis found no statistically significant benefit across all 15 studies.",
    "Controlled trials showed the treatment group performed worse than expected.",
    "The hypothesis was definitively refuted by the experimental evidence.",
    "Independent verification found the original results could not be replicated.",
    "The government report found the policy had no positive effect.",
    "Results were not statistically significant in any of the 8 sub-studies.",
    "The study found the opposite of what was hypothesized.",
    "Three separate labs failed to reproduce the original findings.",
    "The WHO stated the evidence does not support this treatment.",
    "FDA analysis found the drug's benefits did not outweigh its risks.",
    "The trial was stopped early due to lack of efficacy.",
    "No difference was observed between treatment and control groups.",
    "The intervention group had higher mortality rates.",
    "Systematic review found inconsistent and negative results.",
    "Post-market surveillance revealed no clinical benefit.",
    "The proposed mechanism was not supported by experimental data.",
    "Epidemiological data contradicts the causal hypothesis.",
    "The treatment arm showed no improvement over standard care.",
    "Biomarker levels were unchanged despite treatment.",
    "Quality of life scores were identical between groups.",
    "The study failed to demonstrate any meaningful clinical benefit.",
    "No statistically significant difference was found between groups.",
    "The drug did not meet the primary endpoint of the trial.",
    "The intervention had no impact on the primary outcome measure.",
    "The drug was no better than placebo on any endpoint.",
    "The clinical trial showed no efficacy whatsoever.",
    "The data failed to support the proposed hypothesis.",
    "The study found the treatment was completely ineffective.",
    "The drug showed no benefit in any of the tested populations.",
    "No improvement was seen in either treatment or control groups.",
    "The results were uniformly negative across all endpoints.",
    "The intervention was ineffective in preventing disease progression.",
    "The drug failed to demonstrate superiority over placebo.",
    "The study found zero benefit from the intervention.",
    "The drug was shown to be therapeutically useless.",
    "The evidence does not support the use of this treatment.",
    "The clinical trial produced entirely negative results.",
    "The drug had no measurable effect on any outcome.",
    "The intervention did not improve patient outcomes.",
    "The study conclusively demonstrated the drug's lack of efficacy.",
    "The drug was found to be no better than doing nothing.",
    "The treatment had no effect on disease progression.",
    "The experimental results completely failed to support the hypothesis.",
    "The drug was shown to be clinically ineffective.",
    "No clinical benefit was observed with the intervention.",
    "The treatment was no better than observation alone.",
    "The drug failed to meet even a single secondary endpoint.",
    "The study found the drug had zero therapeutic value.",
    "The intervention was entirely without benefit.",
    "The data clearly show the drug does not work.",
    "The clinical evidence contradicts the efficacy claims.",
    "The drug showed no advantage over standard treatment.",
    "The trial was terminated because the drug was not working.",
    "The study found no evidence that the drug is effective.",
    "The drug failed to show any clinical benefit whatsoever.",
    "The intervention had absolutely no impact on outcomes.",
    "The results demonstrated a complete lack of efficacy.",
    "The drug was ineffective across all tested indications.",
    "No benefit was demonstrated in any patient subgroup.",
    "The study showed the drug is clinically worthless.",
    "The drug did not produce any measurable clinical improvement.",
    "The trial results clearly refuted the efficacy hypothesis.",
    "The drug was found to be no better than sugar pills.",
    "The study provided strong evidence against using this drug.",
    "The drug was shown to have no effect on disease outcomes.",
    "The clinical data demonstrate the drug is not effective.",
    "The drug failed to produce any statistically significant improvement.",
    "No evidence of benefit was found in any analysis.",
    "The drug was shown to be ineffective in multiple trials.",
    "The intervention produced no detectable clinical benefit.",
    "The study found the drug does not work as claimed.",
    "The clinical trial demonstrated no efficacy for this drug.",
    "The drug had no measurable effect on the disease.",
    "The evidence firmly contradicts the use of this treatment.",
    "The drug showed no benefit over standard therapy.",
    "The study failed to show any advantage of the treatment.",
    "The drug was completely ineffective in all analyses.",
    "No clinical benefit was demonstrated with this intervention.",
    "The drug was not superior to placebo on any measure.",
    "The evidence clearly shows the drug does not work.",
    "The study found the drug offered no therapeutic benefit.",
    "The drug was therapeutically ineffective in all populations tested.",
    "The clinical trial was a clear failure.",
    "The drug had no positive effect on any outcome.",
    "The study definitively showed the drug is not effective.",
    "The drug failed to demonstrate clinical utility.",
    "No benefit of treatment was observed.",
    "The drug was shown to be without therapeutic value.",
    "The intervention had no effect on patient outcomes.",
    "The drug was not effective in reducing symptoms.",
    "The clinical evidence refutes the claimed benefits.",
    "The drug showed no improvement in any endpoint tested.",
    "The study found the drug to be clinically useless.",
    "The drug did not outperform placebo in any analysis.",
    "The intervention was demonstrated to be completely ineffective.",
    "The drug was no better than no treatment at all.",
    "The study found the drug had no effect on disease.",
    "The drug was proven to be ineffective in large trials.",
    "The clinical data show the drug does not work.",
    "The drug was shown to lack any therapeutic efficacy.",
    "The trial conclusively demonstrated the drug's ineffectiveness.",
    "The drug was found to have zero clinical benefit.",
    "The study provided strong evidence against this drug's efficacy.",
    "The drug was entirely without therapeutic effect.",
    "The evidence demonstrates this drug does not work.",
    "The trial showed the drug is not clinically useful.",
    "The drug was found to be ineffective in all analyses.",
    "The study conclusively showed the drug is ineffective.",
    "The drug had no effect on any measured outcome.",
    "The intervention provided no clinical benefit whatsoever.",
    "The drug was shown to be completely without benefit.",
    "The study failed to demonstrate any efficacy for this drug.",
    "The drug was not effective compared to placebo.",
    "The drug showed no benefit in the clinical trial.",
    "The clinical data showed no evidence of efficacy.",
    "The drug was ineffective in every analysis performed.",
    "The study found the drug offered no clinical advantage.",
    "The drug was shown to have no therapeutic effect.",
    "The clinical trial results were entirely negative.",
    "The drug was ineffective across all study populations.",
    "No improvement was observed with this drug.",
    "The drug was shown to provide no clinical benefit.",
    "The study found the drug was therapeutically useless.",
    "The drug did not produce any clinical improvement.",
    "The trial demonstrated no efficacy for this intervention.",
    "The drug was not effective in treating the condition.",
    "The clinical evidence does not support this drug's use.",
    "The study found the drug was completely ineffective.",
    "The drug had no beneficial effect on any outcome.",
    "The intervention showed no clinical improvement whatsoever.",
    "The drug was demonstrated to be entirely ineffective.",
    "The trial results clearly show the drug does not work.",
    "The drug was found to be without any clinical benefit.",
    "The study showed no evidence of drug efficacy.",
    "The drug was not effective in any tested population.",
    "The drug had no measurable clinical impact.",
    "The clinical trial proved the drug is ineffective.",
    "The evidence does not support the efficacy of this drug.",
    "The study found the drug was clinically worthless.",
    "The drug was ineffective in all tested conditions.",
    "The drug showed no benefit compared to no treatment.",
    "The drug failed to show any clinical improvement.",
    "The clinical evidence firmly contradicts this drug's use.",
    "The drug was not effective in any subgroup analyzed.",
    "The trial demonstrated the drug is therapeutically useless.",
    "The drug was shown to be completely ineffective.",
    "The study found no clinical benefit from this drug.",
    "The drug had zero effect on any measured outcome.",
    "The drug was not effective in preventing disease.",
    "The drug showed no improvement over placebo.",
    "The evidence clearly refutes the drug's efficacy.",
    "The drug was found to have no therapeutic benefit.",
    "The study conclusively showed no benefit from the drug.",
    "The drug was clinically ineffective in every analysis.",
    "The drug was shown to be without any clinical effect.",
    "The trial results conclusively showed the drug is not effective.",
    "The drug was found to be completely without therapeutic value.",
    "The clinical trial showed the drug had no effect.",
    "The drug was not effective compared to the control.",
    "The study demonstrated the drug is therapeutically useless.",
    "The drug had no clinical effect whatsoever.",
    "The drug was shown to be clinically ineffective.",
    "The evidence does not support using this drug.",
    "The trial showed the drug was without benefit.",
    "The drug was not effective in reducing mortality.",
    "The drug was ineffective across all endpoints.",
    "The drug was shown to have no effect on survival.",
    "The drug was clinically useless in all tested indications.",
    "The study found no evidence of drug effectiveness.",
    "The drug was not effective in any analysis.",
    "The drug had no measurable impact on clinical outcomes.",
    "The clinical evidence shows the drug is not effective.",
    "The drug was proven ineffective in rigorous trials.",
]
for s in _refutes:
    EVIDENCE_DATA.append((s, 1))

# NEUTRAL (label 2) — 180+ examples
_neutral = [
    "Results were mixed across different populations and demographics.",
    "Some participants improved while others showed no change.",
    "The effect size was small and borderline significant.",
    "More research is needed to confirm these preliminary findings.",
    "Evidence is insufficient to draw a definitive conclusion.",
    "The results were inconsistent across different study designs.",
    "The study had important limitations that affect interpretation.",
    "Additional large-scale trials are warranted before conclusions.",
    "The effect was observed only in certain subgroups.",
    "Preliminary data suggests a trend but lacks statistical power.",
    "The findings depend heavily on the specific methodology used.",
    "Results varied considerably depending on dosage and duration.",
    "The study population may not be representative of the general public.",
    "Some evidence supports while other evidence contradicts.",
    "The relationship appears complex and not fully understood yet.",
    "The results were inconclusive due to high dropout rates.",
    "Effectiveness varied by geographic region.",
    "Short-term benefits were observed but long-term effects unknown.",
    "The study was underpowered to detect clinically meaningful differences.",
    "Confounding factors may have influenced the results.",
    "The intervention showed promise in early phases but results are preliminary.",
    "Dose-response relationship was not clearly established.",
    "Safety profile was acceptable but efficacy data is limited.",
    "Results from animal studies did not translate to human outcomes.",
    "The evidence is suggestive but not conclusive.",
    "The findings are preliminary and require further validation.",
    "Results from different trials have been inconsistent.",
    "The effect may depend on the patient population studied.",
    "The study was observational, limiting causal inference.",
    "Mixed results were obtained across different outcome measures.",
    "The clinical significance of the findings remains uncertain.",
    "The evidence is equivocal and does not favor either intervention.",
    "The effect size was small and may not be clinically meaningful.",
    "The study was not designed to assess long-term outcomes.",
    "Results were borderline and did not reach significance.",
    "The findings are hypothesis-generating rather than confirmatory.",
    "The evidence is limited by the small sample size.",
    "The relationship between the variables is not well understood.",
    "The study had significant methodological limitations.",
    "Results may vary depending on the specific population studied.",
    "The clinical implications of these findings are unclear.",
    "The data are insufficient to determine efficacy or harm.",
    "The results are promising but require confirmation.",
    "The evidence does not clearly support or refute the claim.",
    "The study provides preliminary data that needs further research.",
    "The findings are inconsistent with prior published literature.",
    "The clinical relevance of the observed effect is uncertain.",
    "The study was too short to assess long-term outcomes.",
    "The results were mixed, making interpretation difficult.",
    "The evidence is insufficient for a strong recommendation.",
    "The study had high risk of bias that may affect results.",
    "The effect was not consistent across different outcome measures.",
    "The findings are conflicting and require additional studies.",
    "The results may not be generalizable to other populations.",
    "The evidence is equivocal regarding clinical benefit.",
    "The study was not adequately powered for the primary outcome.",
    "The results are uncertain and may change with additional data.",
    "The clinical significance of these results is debatable.",
    "The findings require replication in independent cohorts.",
    "The evidence is neither strong enough to support nor refute the intervention.",
    "The study results were ambiguous and difficult to interpret.",
    "The observed effects were modest and of uncertain clinical significance.",
    "The data suggest a possible benefit but further research is needed.",
    "The relationship between the intervention and outcome is unclear.",
    "The study was limited by its short follow-up period.",
    "The findings could be explained by chance or bias.",
    "The evidence is mixed, with both positive and negative studies.",
    "The clinical implications remain uncertain pending further research.",
    "The results are preliminary and should be interpreted with caution.",
    "The effect may be present but the study was underpowered.",
    "The findings are inconsistent and do not provide clear guidance.",
    "The evidence is limited and of low to moderate quality.",
    "The study produced conflicting results across different analyses.",
    "The clinical meaning of the observed differences is uncertain.",
    "The findings suggest a possible benefit but are not definitive.",
    "The evidence does not allow definitive conclusions to be drawn.",
    "The study results were equivocal and inconclusive.",
    "The observed effect was small and of uncertain clinical relevance.",
    "The findings require further investigation before implementation.",
    "The evidence is mixed regarding the effectiveness of the intervention.",
    "The study had methodological limitations that affect interpretation.",
    "The results are promising but the evidence base is limited.",
    "The clinical benefit remains to be established.",
    "The findings are mixed and do not support a clear recommendation.",
    "The study provides limited evidence that needs further investigation.",
    "The results suggest a possible but unproven benefit.",
    "The evidence is too limited to draw firm conclusions.",
    "The findings are preliminary and require validation.",
    "The clinical significance of the results is not established.",
    "The study did not have sufficient power to detect a difference.",
    "The observed benefit may be due to chance.",
    "The findings are ambiguous and require additional research.",
    "The evidence is insufficient to support a strong conclusion.",
    "The results were mixed and interpretation is challenging.",
    "The clinical relevance of the findings is questionable.",
    "The study provided preliminary evidence requiring confirmation.",
    "The effect size was modest and uncertain.",
    "The findings suggest a potential benefit that needs validation.",
    "The evidence is inconclusive and does not support a recommendation.",
    "The study results were equivocal across all endpoints.",
    "The clinical utility of the intervention remains uncertain.",
    "The evidence does not clearly support any conclusion.",
    "The findings may be relevant but require further study.",
    "The study was limited by its observational design.",
    "The results are suggestive but not definitive.",
    "The clinical significance of the findings is not clear.",
    "The evidence is mixed and does not support firm conclusions.",
    "The findings are hypothesis-generating and require further testing.",
    "The study produced ambiguous results that are difficult to interpret.",
    "The observed effects may or may not be clinically meaningful.",
    "The evidence is insufficient to guide clinical decision-making.",
    "The results are uncertain and should be interpreted cautiously.",
    "The clinical implications of the study are unclear.",
    "The findings are preliminary and need confirmation in larger studies.",
    "The evidence base is limited and of variable quality.",
    "The study results were mixed across different populations.",
    "The clinical benefit is uncertain pending further research.",
    "The findings suggest possible but unconfirmed efficacy.",
    "The evidence does not support any definitive conclusion.",
    "The results are inconclusive and require further investigation.",
    "The study had significant limitations that limit interpretation.",
    "The observed effects were of uncertain clinical significance.",
    "The findings may indicate a potential benefit but more research is needed.",
    "The clinical relevance of these findings remains to be established.",
    "The evidence is mixed, with some studies showing benefit and others not.",
    "The results were borderline and require further validation.",
    "The clinical benefit of the intervention is uncertain.",
    "The study provides preliminary evidence that requires confirmation.",
    "The findings are equivocal and do not clearly support treatment.",
    "The evidence is insufficient to determine whether the drug works.",
    "The results suggest a possible benefit but evidence is limited.",
    "The clinical significance of the observed effects is uncertain.",
    "The study was too small to draw definitive conclusions.",
    "The findings are promising but need further confirmation.",
    "The evidence does not allow strong conclusions to be drawn.",
    "The study provided limited evidence of uncertain clinical value.",
    "The results were ambiguous across all endpoints tested.",
    "The clinical utility of the findings is not established.",
    "The evidence is mixed and inconclusive regarding efficacy.",
    "The findings suggest possible benefit but are not definitive.",
    "The study had important limitations that affect its conclusions.",
    "The observed effects were modest and of uncertain significance.",
    "The results require replication before clinical application.",
    "The evidence is insufficient to recommend for or against the intervention.",
    "The findings are preliminary and need further investigation.",
    "The clinical meaning of the results is uncertain.",
    "The study provided limited and inconclusive evidence.",
    "The results are suggestive but not confirmatory.",
    "The evidence is too limited to determine clinical utility.",
    "The findings are mixed and inconclusive.",
    "The clinical implications of the study are unclear.",
    "The evidence does not provide clear guidance for clinical practice.",
    "The results were inconsistent and difficult to interpret.",
    "The observed benefit may not be clinically meaningful.",
    "The findings require additional studies for confirmation.",
    "The evidence is limited and does not support strong conclusions.",
    "The study was underpowered and results are uncertain.",
    "The clinical significance of the findings is debatable.",
    "The evidence is equivocal and does not support any recommendation.",
    "The results suggest possible but unproven clinical benefit.",
    "The study was not designed to answer the primary clinical question.",
    "The findings need validation in larger and more diverse populations.",
    "The evidence does not clearly support or refute efficacy.",
    "The results are preliminary and require further research.",
    "The clinical relevance of the findings has not been established.",
    "The study provided limited evidence that requires further study.",
    "The findings are inconclusive and do not guide clinical practice.",
    "The evidence is insufficient to determine the risk-benefit ratio.",
    "The observed effects were small and uncertain.",
    "The clinical benefit remains to be demonstrated.",
    "The results are promising but the evidence is limited.",
    "The findings are preliminary and should not guide clinical practice.",
    "The evidence is mixed and does not support firm recommendations.",
    "The study produced inconclusive results across all endpoints.",
    "The clinical utility of the intervention needs further study.",
    "The findings are ambiguous and require additional research.",
    "The evidence is insufficient to draw any conclusions.",
    "The results suggest a trend but are not statistically significant.",
    "The study was too short to assess meaningful clinical outcomes.",
    "The clinical implications of the findings remain uncertain.",
    "The evidence is limited and requires further investigation.",
    "The findings need confirmation in independent populations.",
    "The results are uncertain and should be considered preliminary.",
]
for s in _neutral:
    EVIDENCE_DATA.append((s, 2))

print(f"Evidence training data: {len(EVIDENCE_DATA)} examples")


# ════════════════════════════════════════════════════════════════
# MASSIVE TRAINING DATA — Contradiction Detection
# ════════════════════════════════════════════════════════════════

CONTRADICTION_DATA = []

# CONTRADICTION (label 0) — 180+ examples
_contra = [
    ("The meeting is at 3 PM", "The meeting is at 4 PM"),
    ("The drug is effective", "The drug is ineffective"),
    ("All students passed the exam", "Some students failed the exam"),
    ("It is raining outside", "It is sunny and completely dry"),
    ("Revenue increased 15%", "Revenue decreased 15%"),
    ("The company is profitable", "The company reported significant losses"),
    ("Water boils at 100C", "Water boils at 90C at sea level"),
    ("The Earth is approximately round", "The Earth is completely flat"),
    ("Light travels faster than sound", "Sound travels faster than light"),
    ("Cats are mammals", "Cats are reptiles"),
    ("The product costs $50", "The product costs $500"),
    ("The event is on Monday", "The event is on Tuesday"),
    ("Paris is the capital of France", "Lyon is the capital of France"),
    ("The population is 1 million", "The population is 10 million"),
    ("The study supports the hypothesis", "The study contradicts the hypothesis"),
    ("The vaccine is safe", "The vaccine is not safe"),
    ("Climate change is real", "Climate change is not real"),
    ("AI will replace jobs", "AI will create jobs"),
    ("The system is online", "The system is offline"),
    ("The temperature rose to 30C", "The temperature dropped to 10C"),
    ("The company's revenue doubled", "The company's revenue halved"),
    ("Exercise improves health", "Exercise has no effect on health"),
    ("The treatment is beneficial", "The treatment is harmful"),
    ("The algorithm is fast", "The algorithm is very slow"),
    ("The experiment succeeded", "The experiment failed"),
    ("The company grew by 20%", "The company shrank by 20%"),
    ("The patient recovered", "The patient's condition worsened"),
    ("The drug reduces symptoms", "The drug increases symptoms"),
    ("The study showed improvement", "The study showed decline"),
    ("The product is affordable", "The product is extremely expensive"),
    ("The software is reliable", "The software crashes constantly"),
    ("The school has 500 students", "The school has 5000 students"),
    ("The bridge is 100 meters long", "The bridge is 1 kilometer long"),
    ("The test took 1 hour", "The test took 5 hours"),
    ("All employees were promoted", "No employees were promoted"),
    ("The restaurant is open", "The restaurant is permanently closed"),
    ("The car uses gasoline", "The car uses electricity"),
    ("The theory has been proven", "The theory has been disproven"),
    ("The disease is curable", "The disease is incurable"),
    ("The plan was approved", "The plan was rejected"),
    ("The results were positive", "The results were negative"),
    ("The team won the game", "The team lost the game"),
    ("The stock price went up", "The stock price went down"),
    ("The project was completed early", "The project was delayed"),
    ("The medication is effective", "The medication is not effective"),
    ("The building has 10 floors", "The building has 50 floors"),
    ("The experiment took 2 weeks", "The experiment took 2 years"),
    ("The city has 100,000 people", "The city has 1,000,000 people"),
    ("The data supports the claim", "The data refutes the claim"),
    ("The program was successful", "The program was a complete failure"),
    ("The company employs 100 people", "The company employs 10,000 people"),
    ("The lake is 50 meters deep", "The lake is 500 meters deep"),
    ("The volcano is dormant", "The volcano is actively erupting"),
    ("The patient is improving", "The patient is deteriorating rapidly"),
    ("The algorithm runs in O(n)", "The algorithm runs in O(n^2)"),
    ("The country has universal healthcare", "The country has no healthcare system"),
    ("The test results were normal", "The test results were highly abnormal"),
    ("The river flows north", "The river flows south"),
    ("The building was renovated", "The building was demolished"),
    ("The policy increased equality", "The policy increased inequality"),
    ("The drug was FDA approved", "The drug was FDA rejected"),
    ("The product launched in 2020", "The product launched in 2010"),
    ("The company is based in New York", "The company is based in Tokyo"),
    ("The study had 100 participants", "The study had 10,000 participants"),
    ("The experiment was controlled", "The experiment was uncontrolled"),
    ("The vaccine has 95% efficacy", "The vaccine has 30% efficacy"),
    ("The algorithm is deterministic", "The algorithm is random"),
    ("The system uses encryption", "The system has no security"),
    ("The building survived the earthquake", "The building collapsed"),
    ("The treatment costs $100", "The treatment costs $100,000"),
    ("The company is profitable", "The company is bankrupt"),
    ("The study was peer-reviewed", "The study was not peer-reviewed"),
    ("The product is eco-friendly", "The product is highly polluting"),
    ("The disease is spreading", "The disease has been eradicated"),
    ("The software is open source", "The software is proprietary"),
    ("The school is public", "The school is private"),
    ("The patient has no side effects", "The patient has severe side effects"),
    ("The company increased wages", "The company decreased wages"),
    ("The test was easy", "The test was extremely difficult"),
    ("The country is at peace", "The country is at war"),
    ("The team scored 5 goals", "The team scored 0 goals"),
    ("The product is available", "The product is discontinued"),
    ("The city is growing", "The city is declining"),
    ("The experiment was successful", "The experiment failed completely"),
    ("The drug treats the disease", "The drug worsens the disease"),
    ("The company hired 50 people", "The company laid off 50 people"),
    ("The student passed the exam", "The student failed the exam"),
    ("The building is new", "The building is over 100 years old"),
    ("The temperature is 30C", "The temperature is -10C"),
    ("The patient is alive", "The patient died"),
    ("The company won the contract", "The company lost the contract"),
    ("The tree is 10 meters tall", "The tree is 1 meter tall"),
    ("The software update improved performance", "The software update degraded performance"),
    ("The earthquake measured 2.0", "The earthquake measured 8.0"),
    ("The school has a 90% graduation rate", "The school has a 50% graduation rate"),
    ("The car gets 50 mpg", "The car gets 5 mpg"),
    ("The country has free elections", "The country has no elections"),
    ("The project finished under budget", "The project finished way over budget"),
    ("The patient's health improved", "The patient's health declined sharply"),
    ("The company is expanding", "The company is downsizing"),
    ("The experiment proved the theory", "The experiment disproved the theory"),
    ("The drug is taken daily", "The drug is taken once a month"),
    ("The building has solar panels", "The building has no energy systems"),
    ("The lake is clean", "The lake is heavily polluted"),
    ("The species is increasing", "The species is going extinct"),
    ("The patient responded to treatment", "The patient did not respond to treatment"),
    ("The city has low crime", "The city has the highest crime rate"),
    ("The product is lightweight", "The product is extremely heavy"),
    ("The test was comprehensive", "The test was very superficial"),
    ("The company uses renewable energy", "The company uses only fossil fuels"),
    ("The disease is mild", "The disease is fatal"),
    ("The program exceeded expectations", "The program fell short of expectations"),
    ("The student is a fast learner", "The student learns very slowly"),
    ("The building passed inspection", "The building failed inspection"),
    ("The river is clean", "The river is contaminated"),
    ("The company reported a profit", "The company reported a loss"),
    ("The weather is sunny", "The weather is stormy"),
    ("The patient needs surgery", "The patient needs no treatment"),
    ("The school ranks #1", "The school ranks last"),
    ("The product costs less", "The product costs more"),
    ("The algorithm is efficient", "The algorithm is extremely inefficient"),
    ("The system is secure", "The system is vulnerable to attacks"),
    ("The project is on track", "The project is far behind schedule"),
    ("The country is democratic", "The country is authoritarian"),
    ("The vaccine prevents disease", "The vaccine causes disease"),
    ("The company is innovative", "The company is stuck in the past"),
    ("The patient recovered fully", "The patient never recovered"),
    ("The tree is healthy", "The tree is dying"),
    ("The experiment confirmed the hypothesis", "The experiment rejected the hypothesis"),
    ("The drug was well-tolerated", "The drug caused severe adverse effects"),
    ("The city is affordable", "The city is unaffordable"),
    ("The company values transparency", "The company operates in secrecy"),
    ("The results were significant", "The results were not significant"),
    ("The treatment is first-line", "The treatment is last-resort"),
    ("The patient is stable", "The patient is in critical condition"),
    ("The school is well-funded", "The school is severely underfunded"),
    ("The product is durable", "The product breaks easily"),
    ("The study was randomized", "The study was biased"),
    ("The company is growing fast", "The company is shrinking rapidly"),
    ("The patient has no allergies", "The patient is allergic to everything"),
    ("The river is shallow", "The river is extremely deep"),
    ("The algorithm is correct", "The algorithm has bugs"),
    ("The building is energy efficient", "The building wastes energy"),
    ("The software is user-friendly", "The software is impossible to use"),
    ("The country has high literacy", "The country has very low literacy"),
    ("The company is ethical", "The company is corrupt"),
    ("The patient is young", "The patient is elderly"),
    ("The test was fair", "The test was completely biased"),
    ("The treatment works for everyone", "The treatment works for no one"),
    ("The city is safe", "The city is extremely dangerous"),
    ("The product is well-reviewed", "The product has terrible reviews"),
    ("The study was large", "The study was tiny"),
    ("The company is transparent", "The company hides information"),
    ("The patient is getting better", "The patient is getting worse"),
    ("The school is prestigious", "The school is unknown"),
    ("The drug is cheap", "The drug is very expensive"),
    ("The results were clear", "The results were ambiguous"),
    ("The project succeeded", "The project failed miserably"),
    ("The company is reliable", "The company is unreliable"),
    ("The algorithm is fast", "The algorithm takes forever"),
    ("The system is working", "The system is completely broken"),
    ("The study is rigorous", "The study is poorly designed"),
    ("The treatment is painless", "The treatment is extremely painful"),
    ("The patient is conscious", "The patient is unconscious"),
    ("The building is accessible", "The building is inaccessible"),
    ("The product is available everywhere", "The product is impossible to find"),
    ("The country is rich", "The country is extremely poor"),
    ("The company is inclusive", "The company is exclusive"),
    ("The test was completed", "The test was never started"),
    ("The results were accurate", "The results were completely wrong"),
    ("The patient is thriving", "The patient is suffering"),
    ("The school is improving", "The school is declining rapidly"),
    ("The drug is effective immediately", "The drug has no immediate effect"),
    ("The project is innovative", "The project is outdated"),
    ("The company is stable", "The company is on the verge of collapse"),
    ("The algorithm scales well", "The algorithm does not scale"),
    ("The system is responsive", "The system is unresponsive"),
    ("The study is groundbreaking", "The study is unremarkable"),
    ("The treatment cures the disease", "The treatment has no effect on the disease"),
    ("The patient is healthy", "The patient is severely ill"),
    ("The product is safe", "The product is dangerous"),
    ("The company is thriving", "The company is failing"),
    ("The results are encouraging", "The results are discouraging"),
    ("The project is ahead of schedule", "The project is months behind"),
    ("The country is peaceful", "The country is in chaos"),
    ("The algorithm is optimal", "The algorithm is suboptimal"),
    ("The system is stable", "The system is unstable"),
    ("The study is conclusive", "The study is inconclusive"),
]
for a, b in _contra:
    CONTRADICTION_DATA.append((a, b, 0))

# CONSISTENT (label 1) — 180+ examples
_consistent = [
    ("The study found exercise improves health", "Research confirms physical activity benefits cardiovascular health"),
    ("Water boils at 100C at sea level", "The boiling point of water is 100 degrees Celsius"),
    ("Paris is the capital of France", "France's capital city is Paris"),
    ("The experiment showed positive results", "The trial demonstrated beneficial outcomes"),
    ("The company reported growth", "The organization announced expansion"),
    ("The medication reduces symptoms", "The drug alleviates clinical symptoms"),
    ("Climate change is accelerating", "Global warming trends are intensifying"),
    ("The algorithm runs in O(n log n)", "The algorithm has logarithmic linear time complexity"),
    ("The bridge is 500 meters long", "The 500m bridge spans the river"),
    ("Python is a programming language", "Python is used for software development"),
    ("The study confirmed the hypothesis", "Research validated the proposed theory"),
    ("The system is operational", "The system is online and functioning"),
    ("The project completed on time", "The project finished ahead of schedule"),
    ("The drug reduces inflammation", "The medication decreases inflammatory response"),
    ("The company is profitable", "The organization reported positive earnings"),
    ("The algorithm is efficient", "The method performs well computationally"),
    ("The treatment improved outcomes", "The intervention enhanced patient results"),
    ("The study was large-scale", "The research involved a substantial cohort"),
    ("The results were significant", "The findings reached statistical significance"),
    ("The drug is safe", "The medication has acceptable safety profile"),
    ("The intervention worked", "The treatment was effective"),
    ("The data supports the claim", "Evidence corroborates the assertion"),
    ("The hypothesis was confirmed", "The theory was validated"),
    ("The system works", "The system is functional"),
    ("The experiment was successful", "The trial achieved its endpoints"),
    ("The patient improved after treatment", "The patient showed recovery following therapy"),
    ("The vaccine is 95% effective", "The immunization provides 95% protection"),
    ("The company has 1000 employees", "The organization employs one thousand people"),
    ("The temperature is 25C", "It is 25 degrees Celsius outside"),
    ("The car travels at 60 mph", "The vehicle moves at 60 miles per hour"),
    ("The test was easy", "The exam was simple"),
    ("The patient is recovering", "The patient is getting better"),
    ("The software is fast", "The application performs quickly"),
    ("The building is tall", "The structure is high-rise"),
    ("The river is long", "The waterway extends for many miles"),
    ("The study was comprehensive", "The research was thorough"),
    ("The product is popular", "The item is widely used"),
    ("The company is growing", "The business is expanding"),
    ("The disease is treatable", "The illness can be treated"),
    ("The experiment was rigorous", "The study was well-controlled"),
    ("The school is excellent", "The educational institution is outstanding"),
    ("The drug is approved", "The medication has regulatory approval"),
    ("The algorithm is correct", "The method produces accurate results"),
    ("The system is secure", "The platform has strong protection"),
    ("The treatment is effective", "The therapy works well"),
    ("The patient is stable", "The patient's condition is steady"),
    ("The company is innovative", "The organization is creative"),
    ("The results are consistent", "The findings are reproducible"),
    ("The project is successful", "The initiative achieved its goals"),
    ("The study is well-designed", "The research has solid methodology"),
    ("The product is high-quality", "The item is well-made"),
    ("The software is reliable", "The application is dependable"),
    ("The city is beautiful", "The town is attractive"),
    ("The country is developed", "The nation is industrialized"),
    ("The patient has no complications", "The patient experienced no issues"),
    ("The company reported a profit", "The business posted earnings"),
    ("The test was comprehensive", "The examination was thorough"),
    ("The study showed benefit", "The research demonstrated positive effects"),
    ("The drug works well", "The medication is effective"),
    ("The system is efficient", "The platform operates smoothly"),
    ("The project is on track", "The initiative is progressing well"),
    ("The algorithm is optimized", "The method has been streamlined"),
    ("The patient responded to treatment", "The patient benefited from therapy"),
    ("The company is stable", "The business is steady"),
    ("The results were positive", "The findings were encouraging"),
    ("The treatment helped patients", "The therapy improved patient outcomes"),
    ("The study was well-conducted", "The research was properly executed"),
    ("The product is approved", "The item has been cleared"),
    ("The software is updated", "The application has been patched"),
    ("The disease is rare", "The condition is uncommon"),
    ("The patient is healthy", "The individual is well"),
    ("The company is profitable", "The firm is making money"),
    ("The experiment worked", "The trial was successful"),
    ("The algorithm is fast", "The method is quick"),
    ("The system is online", "The platform is available"),
    ("The study found a benefit", "The research detected improvement"),
    ("The drug is prescribed", "The medication is recommended"),
    ("The patient recovered quickly", "The patient healed rapidly"),
    ("The company is growing", "The organization is expanding"),
    ("The results are clear", "The findings are unambiguous"),
    ("The treatment is standard", "The therapy is conventional"),
    ("The project was completed", "The initiative was finished"),
    ("The software is user-friendly", "The application is easy to use"),
    ("The school is well-funded", "The institution has adequate resources"),
    ("The city is safe", "The town has low crime"),
    ("The country has elections", "The nation holds regular votes"),
    ("The patient is asymptomatic", "The patient shows no symptoms"),
    ("The company is ethical", "The firm operates with integrity"),
    ("The study was peer-reviewed", "The research was independently evaluated"),
    ("The product is sustainable", "The item is environmentally friendly"),
    ("The algorithm is scalable", "The method handles growth well"),
    ("The system is reliable", "The platform is dependable"),
    ("The treatment is evidence-based", "The therapy is research-supported"),
    ("The patient is improving", "The individual is getting better"),
    ("The company is transparent", "The firm is open about operations"),
    ("The results are statistically significant", "The findings meet significance thresholds"),
    ("The study was blinded", "The research used blinding"),
    ("The drug has few side effects", "The medication has minimal adverse effects"),
    ("The project met its deadline", "The initiative finished on time"),
    ("The software is secure", "The application has strong security"),
    ("The school is accredited", "The institution is certified"),
    ("The city is growing", "The town is developing"),
    ("The country is stable", "The nation is politically stable"),
    ("The patient tolerates the treatment", "The individual handles therapy well"),
    ("The company is competitive", "The firm performs well in the market"),
    ("The study confirmed safety", "The research validated the safety profile"),
    ("The product is effective", "The item works as intended"),
    ("The algorithm is accurate", "The method produces correct results"),
    ("The system is fast", "The platform responds quickly"),
    ("The treatment is safe", "The therapy has a good safety record"),
    ("The patient has no allergies", "The individual has no allergic reactions"),
    ("The company is reputable", "The firm has a good standing"),
    ("The results are reliable", "The findings are trustworthy"),
    ("The project exceeded expectations", "The initiative surpassed goals"),
    ("The software is compatible", "The application works on multiple platforms"),
    ("The school is diverse", "The institution is multicultural"),
    ("The city is clean", "The town is well-maintained"),
    ("The country is democratic", "The nation has free elections"),
    ("The patient is responsive", "The individual reacts to treatment"),
    ("The company is diverse", "The firm is inclusive"),
    ("The study was randomized", "The research used randomization"),
    ("The product is durable", "The item is long-lasting"),
    ("The algorithm is robust", "The method handles edge cases"),
    ("The system is scalable", "The platform handles growth"),
    ("The treatment is personalized", "The therapy is tailored"),
    ("The patient is compliant", "The individual follows the regimen"),
    ("The company is profitable", "The firm generates revenue"),
    ("The results are reproducible", "The findings can be replicated"),
    ("The project was successful", "The initiative achieved results"),
    ("The software is intuitive", "The application is easy to navigate"),
    ("The school is well-ranked", "The institution is highly rated"),
    ("The city is vibrant", "The town is lively"),
    ("The country is prosperous", "The nation is economically strong"),
    ("The patient is progressing", "The individual is advancing"),
    ("The company is successful", "The firm is doing well"),
    ("The study showed efficacy", "The research demonstrated effectiveness"),
    ("The product is popular", "The item is in high demand"),
    ("The algorithm is correct", "The method gives right answers"),
    ("The system is functional", "The platform works properly"),
    ("The treatment works", "The therapy is effective"),
    ("The patient is stable", "The individual is in good condition"),
    ("The company is growing", "The firm is expanding operations"),
    ("The results are encouraging", "The findings are positive"),
    ("The project was delivered", "The initiative was completed"),
    ("The software works well", "The application functions properly"),
    ("The school is excellent", "The institution provides quality education"),
    ("The city is thriving", "The town is doing well"),
    ("The country is developed", "The nation is advanced"),
    ("The patient is healthy", "The individual is in good health"),
    ("The company is successful", "The firm is achieving its goals"),
    ("The study was thorough", "The research was comprehensive"),
    ("The product is high-quality", "The item meets standards"),
    ("The algorithm is fast", "The method is efficient"),
    ("The system is reliable", "The platform is dependable"),
    ("The treatment is effective", "The therapy produces results"),
    ("The patient improved", "The individual got better"),
    ("The company is growing", "The firm is increasing in size"),
    ("The results are clear", "The findings are obvious"),
    ("The project succeeded", "The initiative was a success"),
    ("The software is great", "The application is excellent"),
    ("The school is good", "The institution is well-regarded"),
    ("The city is nice", "The town is pleasant"),
    ("The country is stable", "The nation is secure"),
    ("The patient is well", "The individual is fine"),
    ("The company is doing well", "The firm is performing well"),
    ("The study confirmed the theory", "The research validated the hypothesis"),
    ("The product is recommended", "The item is endorsed"),
    ("The algorithm is optimized", "The method is refined"),
    ("The system is working", "The platform is operational"),
    ("The treatment is prescribed", "The therapy is recommended"),
    ("The patient is better", "The individual has improved"),
]
for a, b in _consistent:
    CONTRADICTION_DATA.append((a, b, 1))

# NEUTRAL (label 2) — 150+ examples
_contra_neutral = [
    ("The study had mixed results", "Results varied by demographic group"),
    ("Some participants improved", "Effectiveness was inconsistent across subjects"),
    ("The effect size was small", "The magnitude of change was modest"),
    ("More research is needed", "Additional studies are warranted"),
    ("Evidence is insufficient", "Data is lacking for definitive conclusions"),
    ("The results were inconclusive", "Findings were not definitive"),
    ("The study had limitations", "The research had methodological constraints"),
    ("The effect was observed in some conditions", "Benefits were context-dependent"),
    ("Preliminary data suggests a trend", "Early results indicate a potential pattern"),
    ("The findings are complex", "The relationship is multifaceted"),
    ("Results were inconsistent", "Outcomes varied across studies"),
    ("The study was underpowered", "Sample size was insufficient"),
    ("Confounding factors may exist", "Potential confounders were identified"),
    ("Short-term effects were observed", "Immediate outcomes were noted"),
    ("The evidence is mixed", "Data presents contradictory signals"),
    ("The drug has potential benefits", "The medication may offer some advantages"),
    ("The algorithm has trade-offs", "The method involves performance considerations"),
    ("The patient has some symptoms", "The individual reports mild issues"),
    ("The company faces challenges", "The firm encounters obstacles"),
    ("The study provides preliminary data", "The research offers initial findings"),
    ("The treatment shows promise", "The therapy appears potentially useful"),
    ("The product has features", "The item offers various capabilities"),
    ("The software needs updates", "The application requires improvements"),
    ("The school has strengths", "The institution has advantages"),
    ("The city has resources", "The town has available assets"),
    ("The country is developing", "The nation is progressing"),
    ("The patient has a condition", "The individual has a diagnosis"),
    ("The company has goals", "The firm has objectives"),
    ("The results need interpretation", "The findings require analysis"),
    ("The project has phases", "The initiative involves stages"),
    ("The algorithm has variants", "The method has different versions"),
    ("The system has components", "The platform has parts"),
    ("The treatment has options", "The therapy has alternatives"),
    ("The patient needs monitoring", "The individual requires follow-up"),
    ("The company has policies", "The firm has guidelines"),
    ("The study has scope", "The research has boundaries"),
    ("The product has specifications", "The item has details"),
    ("The software has requirements", "The application has needs"),
    ("The school has programs", "The institution offers courses"),
    ("The city has services", "The town provides amenities"),
    ("The country has policies", "The nation has regulations"),
    ("The patient has history", "The individual has background"),
    ("The company has strategy", "The firm has a plan"),
    ("The results suggest trends", "The findings indicate patterns"),
    ("The project has milestones", "The initiative has checkpoints"),
    ("The algorithm has complexity", "The method has computational cost"),
    ("The system has architecture", "The platform has design"),
    ("The treatment has protocols", "The therapy has procedures"),
    ("The patient has needs", "The individual has requirements"),
    ("The company has culture", "The firm has values"),
    ("The study has implications", "The research has consequences"),
    ("The product has limitations", "The item has restrictions"),
    ("The software has bugs", "The application has issues"),
    ("The school has challenges", "The institution faces difficulties"),
    ("The city has issues", "The town has problems"),
    ("The country has concerns", "The nation has matters"),
    ("The patient has concerns", "The individual has worries"),
    ("The company has opportunities", "The firm has possibilities"),
    ("The results raise questions", "The findings prompt inquiries"),
    ("The project has risks", "The initiative has hazards"),
    ("The algorithm has edge cases", "The method has special scenarios"),
    ("The system has vulnerabilities", "The platform has weaknesses"),
    ("The treatment has side effects", "The therapy has adverse reactions"),
    ("The patient has preferences", "The individual has choices"),
    ("The company has competitors", "The firm has rivals"),
    ("The study has findings", "The research has discoveries"),
    ("The product has uses", "The item has applications"),
    ("The software has features", "The application has capabilities"),
    ("The school has faculty", "The institution has teachers"),
    ("The city has neighborhoods", "The town has districts"),
    ("The country has regions", "The nation has areas"),
    ("The patient has options", "The individual has choices"),
    ("The company has assets", "The firm has resources"),
    ("The results are preliminary", "The findings are early"),
    ("The project is ongoing", "The initiative is in progress"),
    ("The algorithm needs refinement", "The method requires improvement"),
    ("The system needs maintenance", "The platform requires upkeep"),
    ("The treatment is being studied", "The therapy is under investigation"),
    ("The patient is being evaluated", "The individual is being assessed"),
    ("The company is adapting", "The firm is adjusting"),
    ("The study is continuing", "The research is ongoing"),
    ("The product is evolving", "The item is being updated"),
    ("The software is developing", "The application is progressing"),
    ("The school is reforming", "The institution is changing"),
    ("The city is transitioning", "The town is evolving"),
    ("The country is reforming", "The nation is changing"),
    ("The patient is adapting", "The individual is adjusting"),
    ("The company is evolving", "The firm is transforming"),
    ("The results are emerging", "The findings are developing"),
    ("The project is developing", "The initiative is advancing"),
    ("The algorithm is being refined", "The method is being improved"),
    ("The system is being upgraded", "The platform is being enhanced"),
    ("The treatment is experimental", "The therapy is novel"),
    ("The patient is participating", "The individual is involved"),
    ("The company is restructuring", "The firm is reorganizing"),
    ("The study is underway", "The research is in progress"),
    ("The product is in development", "The item is being created"),
    ("The software is in beta", "The application is being tested"),
    ("The school is innovating", "The institution is creating"),
    ("The city is planning", "The town is preparing"),
    ("The country is investing", "The nation is funding"),
    ("The patient is learning", "The individual is gaining knowledge"),
    ("The company is researching", "The firm is investigating"),
    ("The results need validation", "The findings need confirmation"),
    ("The project needs approval", "The initiative needs authorization"),
    ("The algorithm needs testing", "The method needs evaluation"),
    ("The system needs review", "The platform needs assessment"),
    ("The treatment needs trials", "The therapy needs studies"),
    ("The patient needs consultation", "The individual needs advice"),
    ("The company needs analysis", "The firm needs examination"),
    ("The study needs replication", "The research needs reproduction"),
    ("The product needs feedback", "The item needs input"),
    ("The software needs debugging", "The application needs fixing"),
    ("The school needs support", "The institution needs assistance"),
    ("The city needs investment", "The town needs funding"),
    ("The country needs reform", "The nation needs change"),
    ("The patient needs time", "The individual needs patience"),
    ("The company needs leadership", "The firm needs direction"),
    ("The results are mixed", "The findings are mixed"),
    ("The project is complex", "The initiative is complicated"),
    ("The algorithm is versatile", "The method is adaptable"),
    ("The system is flexible", "The platform is adaptable"),
    ("The treatment is available", "The therapy is accessible"),
    ("The patient is hopeful", "The individual is optimistic"),
    ("The company is resilient", "The firm is adaptable"),
    ("The study is ongoing", "The research continues"),
    ("The product is versatile", "The item is multifunctional"),
    ("The software is cross-platform", "The application works everywhere"),
    ("The school is inclusive", "The institution welcomes all"),
    ("The city is diverse", "The town is multicultural"),
    ("The country is multicultural", "The nation is diverse"),
    ("The patient is engaged", "The individual is involved"),
    ("The company is collaborative", "The firm is cooperative"),
    ("The results are being analyzed", "The findings are being reviewed"),
    ("The project is being planned", "The initiative is being designed"),
    ("The algorithm is being tested", "The method is being evaluated"),
    ("The system is being designed", "The platform is being built"),
    ("The treatment is being developed", "The therapy is being created"),
    ("The patient is being monitored", "The individual is being watched"),
    ("The company is being evaluated", "The firm is being assessed"),
    ("The study is being conducted", "The research is being carried out"),
    ("The product is being designed", "The item is being crafted"),
    ("The software is being written", "The application is being coded"),
    ("The school is being assessed", "The institution is being reviewed"),
    ("The city is being studied", "The town is being examined"),
    ("The country is being analyzed", "The nation is being evaluated"),
    ("The patient is being treated", "The individual is receiving care"),
    ("The company is being audited", "The firm is being inspected"),
    ("The results are preliminary", "The findings are initial"),
    ("The project is in development", "The initiative is being created"),
    ("The algorithm is experimental", "The method is novel"),
    ("The system is being maintained", "The platform is being managed"),
    ("The treatment is being monitored", "The therapy is being tracked"),
    ("The patient is being followed", "The individual is being observed"),
    ("The company is being managed", "The firm is being run"),
]
for a, b in _contra_neutral:
    CONTRADICTION_DATA.append((a, b, 2))

print(f"Contradiction training data: {len(CONTRADICTION_DATA)} examples")


# ════════════════════════════════════════════════════════════════
# MASSIVE TRAINING DATA — Intent Classification
# ════════════════════════════════════════════════════════════════

INTENT_DATA = []

# investigation (0) — 30+ examples
_investigation = [
    "Investigate John Smith and his connections",
    "Find everything about Alice Chen's background",
    "Who is Bob Wilson and what do they do?",
    "Research Sarah Davis's professional history",
    "Look into TechCorp Inc's public records",
    "Trace the financial history of this company",
    "Map the network of associates for this person",
    "Find all public records for James Taylor",
    "Research the background of this organization",
    "Investigate the history of this building",
    "Who are the key people behind this company?",
    "Find the professional history of Dr. Smith",
    "Research the criminal record of this individual",
    "Look into the ownership structure of this firm",
    "Investigate the source of these funds",
    "Find all connections between these two people",
    "Research the organizational chart of this company",
    "Trace the origin of this product",
    "Investigate the regulatory history of this drug",
    "Find the complete profile of this suspect",
    "Research the academic history of this professor",
    "Look into the patent portfolio of this company",
    "Investigate the supply chain of this product",
    "Find the employment history of this candidate",
    "Research the legal history of this entity",
    "Trace the ownership of this property",
    "Investigate the environmental record of this factory",
    "Find all information about this organization",
    "Research the reputation of this business",
    "Look into the compliance history of this firm",
]
for q in _investigation:
    INTENT_DATA.append((q, 0))

# search (1) — 30+ examples
_search = [
    "Search for information about quantum computing",
    "Find articles about climate change",
    "What's the latest news on AI?",
    "Look up machine learning on the web",
    "Research renewable energy across multiple sources",
    "Search for recent papers on gene therapy",
    "Find reports about the economic situation",
    "What do sources say about this topic?",
    "Find academic papers about deep learning",
    "Search for data on global warming",
    "Look for information about this technology",
    "Find articles published this year on this topic",
    "Search for studies on this medication",
    "What are the latest developments in this field?",
    "Find news reports about this event",
    "Search for reviews of this product",
    "Look up statistics on this topic",
    "Find documentation for this software",
    "Search for case studies on this approach",
    "Find the latest research on this disease",
    "Look for expert opinions on this matter",
    "Find all available information on this topic",
    "Search for precedent cases on this issue",
    "What does the literature say about this topic?",
    "Find competing products in this market",
    "Search for patent filings in this area",
    "Find regulatory filings for this company",
    "Look for conference proceedings on this topic",
    "Search for white papers on this technology",
    "Find the latest standards in this industry",
]
for q in _search:
    INTENT_DATA.append((q, 1))

# identity_analysis (2) — 30+ examples
_identity = [
    "Is John Smith the same person as John Doe?",
    "Could Alice C be an alias for Alice Chen?",
    "Analyze the identity of Bob Wilson",
    "Verify if Sarah Davis matches the description",
    "Are these two profiles about the same person?",
    "Does this person match the known identity?",
    "Is there evidence these are the same individual?",
    "Cross-reference this person with known aliases",
    "Can you confirm this is the right person?",
    "Does the biographical data match?",
    "Are these records from the same person?",
    "Is James Taylor connected to JohnS?",
    "Verify if this name matches the suspect",
    "Does this identity match the database entry?",
    "Are these two people the same individual?",
    "Check if this person uses multiple names",
    "Does the photo match the description?",
    "Is this the correct person of interest?",
    "Verify identity consistency across sources",
    "Does this profile match the known person?",
    "Are these two records about one person?",
    "Is there a link between these identities?",
    "Can you verify this person's identity?",
    "Does this individual match the profile?",
    "Are the biographical details consistent?",
    "Is this person who they claim to be?",
    "Does this identity check out?",
    "Verify if these records belong together",
    "Are the identifying details a match?",
    "Cross-check this identity against records",
]
for q in _identity:
    INTENT_DATA.append((q, 2))

# evidence_analysis (3) — 30+ examples
_evidence = [
    "What evidence supports climate change?",
    "Is there proof that the vaccine works?",
    "Evaluate the evidence for this claim",
    "What do the sources say about this topic?",
    "How strong is the evidence?",
    "Review the evidence for this treatment",
    "What is the quality of evidence available?",
    "Assess the strength of the evidence base",
    "Is there sufficient evidence to support this?",
    "What does the research evidence show?",
    "Evaluate the supporting documentation",
    "How reliable is the evidence provided?",
    "What are the key pieces of evidence?",
    "Assess the credibility of the evidence",
    "Is the evidence convincing?",
    "What evidence contradicts this claim?",
    "How much evidence exists for this theory?",
    "Evaluate the evidence quality",
    "What is the evidence-based conclusion?",
    "Does the evidence support or refute this?",
    "Review all available evidence on this matter",
    "What do clinical trials show?",
    "How does the evidence weigh up?",
    "Is there enough evidence to decide?",
    "What does the scientific evidence say?",
    "Assess the totality of the evidence",
    "Is this claim supported by evidence?",
    "What evidence is available from studies?",
    "Evaluate the proof for this assertion",
    "How strong is the case for this theory?",
]
for q in _evidence:
    INTENT_DATA.append((q, 3))

# comparison (4) — 30+ examples
_comparison = [
    "Compare Python and JavaScript",
    "What are the differences between SQL and NoSQL?",
    "How does React differ from Vue?",
    "Compare the evidence for A vs B",
    "Which is more reliable: source A or source B?",
    "What are the similarities and differences?",
    "How do these two approaches compare?",
    "Compare the pros and cons of each option",
    "Which treatment is better for this condition?",
    "How do these products stack up against each other?",
    "Compare the performance metrics of both systems",
    "What distinguishes these two methods?",
    "How do the costs compare between these options?",
    "Which study design is more rigorous?",
    "Compare the effectiveness of these interventions",
    "What are the trade-offs between these choices?",
    "How does option A compare to option B?",
    "Compare the safety profiles of these drugs",
    "Which source is more credible?",
    "How do these algorithms differ in performance?",
    "Compare the features of these two platforms",
    "What makes these approaches different?",
    "Which methodology is more appropriate?",
    "Compare the outcomes of both studies",
    "How do these results compare to previous research?",
    "Which strategy is more effective?",
    "Compare the risks and benefits of each approach",
    "How do these findings differ from earlier work?",
    "What are the key differences between these options?",
    "Which intervention produces better results?",
]
for q in _comparison:
    INTENT_DATA.append((q, 4))

# timeline (5) — 30+ examples
_timeline = [
    "Create a timeline of World War II",
    "What happened first: the Moon landing or WWII?",
    "When did the Internet originate?",
    "Reconstruct the chronological order of events",
    "What events led to the French Revolution?",
    "Build a timeline of this company's history",
    "What was the sequence of events?",
    "When did these events occur relative to each other?",
    "Create a chronological overview of this period",
    "What is the order of these historical events?",
    "Map out the key dates in this timeline",
    "When was each milestone achieved?",
    "What events happened in this time period?",
    "Create a history of this technology",
    "When did the major changes occur?",
    "What is the chronology of this project?",
    "How did events unfold over time?",
    "What happened during this period?",
    "When were the key developments?",
    "Create a historical timeline of this subject",
    "What was the progression of events?",
    "When did each phase begin and end?",
    "What is the order of these discoveries?",
    "Map the timeline of this research",
    "When did the crisis begin and end?",
    "What events preceded this outcome?",
    "Reconstruct the sequence of decisions",
    "When did the major changes take place?",
    "What is the timeline of this disease?",
    "How did this situation develop over time?",
]
for q in _timeline:
    INTENT_DATA.append((q, 5))

# relationship_analysis (6) — 30+ examples
_relationship = [
    "What is the relationship between Google and Alphabet?",
    "How are these two companies connected?",
    "Map the relationships between these entities",
    "What connections exist between these organizations?",
    "Are these two people affiliated?",
    "What is the link between these events?",
    "How does this person relate to that company?",
    "What is the organizational relationship here?",
    "Are these entities part of the same group?",
    "What is the causal relationship between these events?",
    "How are these departments connected?",
    "What is the supply chain relationship?",
    "Are these studies related to each other?",
    "What is the partnership between these firms?",
    "How does this gene relate to that disease?",
    "What connections exist in this network?",
    "Are these issues interconnected?",
    "What is the correlation between these variables?",
    "How do these factors relate to each other?",
    "What is the dependency between these systems?",
    "Are these projects part of the same initiative?",
    "What is the relationship between cause and effect here?",
    "How are these findings connected?",
    "What is the link between these organizations?",
    "Are these two policies related?",
    "What is the hierarchical relationship?",
    "How do these concepts relate to each other?",
    "What is the connection between these datasets?",
    "Are these results correlated?",
    "What is the relationship between input and output?",
]
for q in _relationship:
    INTENT_DATA.append((q, 6))

# location_analysis (7) — 30+ examples
_location = [
    "Where is Google headquartered?",
    "What is the geographic relationship between NY and LA?",
    "How far is London from Paris?",
    "Map all locations associated with this company",
    "What locations are connected to this event?",
    "Where were these events reported?",
    "What is the geographic distribution?",
    "Which regions are affected?",
    "Where does this phenomenon occur?",
    "What is the spatial pattern here?",
    "Where are the main operations located?",
    "What areas does this organization cover?",
    "Where were the studies conducted?",
    "What is the geographic scope of this issue?",
    "Which countries are involved?",
    "Where is the headquarters located?",
    "What are the branch office locations?",
    "Where did this event take place?",
    "What is the location of the facility?",
    "Which geographic areas are most affected?",
    "Where are the manufacturing plants?",
    "What is the regional breakdown?",
    "Where were the samples collected?",
    "Which cities are involved in this study?",
    "What is the geographic distribution of cases?",
    "Where are the key facilities located?",
    "What locations should be investigated?",
    "Which areas showed the highest impact?",
    "Where are the distribution centers?",
    "What is the territorial coverage?",
]
for q in _location:
    INTENT_DATA.append((q, 7))

# source_verification (8) — 30+ examples
_source = [
    "Is this source reliable?",
    "Verify the credibility of this website",
    "Is this information trustworthy?",
    "What is the reliability of this report?",
    "Check if this is a primary or secondary source",
    "Can this source be trusted?",
    "What is the authority of this publication?",
    "Is this a reputable source?",
    "How credible is this data?",
    "Evaluate the trustworthiness of this source",
    "Is this information from a reliable source?",
    "What is the track record of this organization?",
    "Is this peer-reviewed or not?",
    "What is the source's reputation?",
    "Can this data be verified?",
    "Is this a biased or unbiased source?",
    "What is the publication's impact factor?",
    "Is this a government or commercial source?",
    "How current is this information?",
    "Is this from an expert in the field?",
    "What is the funding source for this study?",
    "Is this information independently verified?",
    "What is the editorial policy of this source?",
    "Is this data from a controlled study?",
    "Can this claim be independently verified?",
    "What is the methodology of this source?",
    "Is this anecdotal or evidence-based?",
    "How does this source compare to others?",
    "Is this from a conflict-of-interest free source?",
    "What are the limitations of this source?",
]
for q in _source:
    INTENT_DATA.append((q, 8))

# contradiction_analysis (9) — 30+ examples
_contra_intent = [
    "Do these statements contradict each other?",
    "Is there a conflict between these claims?",
    "Are these sources consistent?",
    "Find contradictions in the evidence",
    "Do these two reports disagree?",
    "Are these findings consistent with each other?",
    "Is there a discrepancy between these accounts?",
    "Do these data points conflict?",
    "Are these conclusions mutually exclusive?",
    "Is there tension between these findings?",
    "Do these studies agree or disagree?",
    "Are there any inconsistencies in this evidence?",
    "Does this source contradict the other?",
    "Are these two narratives compatible?",
    "Is there a logical conflict here?",
    "Do these facts align with each other?",
    "Are these claims in conflict?",
    "Is there evidence of contradiction?",
    "Do these results contradict each other?",
    "Are these two pieces of information consistent?",
    "Does one source negate the other?",
    "Is there conflicting information here?",
    "Are these accounts mutually consistent?",
    "Do these findings create a paradox?",
    "Is there a logical inconsistency?",
    "Do these data contradict the hypothesis?",
    "Are these results reproducible across studies?",
    "Does the evidence present a unified picture?",
    "Is there agreement between the sources?",
    "Do these facts support or contradict each other?",
]
for q in _contra_intent:
    INTENT_DATA.append((q, 9))

# summarization (10) — 30+ examples
_summary = [
    "Summarize the evidence about this topic",
    "Give me a brief overview of quantum computing",
    "What are the key findings about climate change?",
    "Provide a summary of the investigation",
    "What are the main points?",
    "Summarize the key takeaways",
    "Give a brief of this report",
    "What are the highlights of this study?",
    "Provide an executive summary",
    "What is the gist of this document?",
    "Summarize the main conclusions",
    "Give me a quick overview",
    "What are the essential points?",
    "Summarize the research findings",
    "What are the key messages?",
    "Provide a synopsis of this paper",
    "Summarize the important results",
    "What are the core findings?",
    "Give me a summary of this analysis",
    "What are the bottom-line results?",
    "Summarize the critical points",
    "What should I know about this topic?",
    "Provide a high-level overview",
    "Summarize the evidence base",
    "What are the main conclusions?",
    "Give me a brief summary",
    "What are the key recommendations?",
    "Summarize the report's findings",
    "What are the primary results?",
    "Provide a concise summary",
]
for q in _summary:
    INTENT_DATA.append((q, 10))

# extraction (11) — 30+ examples
_extraction = [
    "Extract all names mentioned in this text",
    "Find all dates in this document",
    "What organizations are mentioned?",
    "Extract all locations from this text",
    "Find all email addresses",
    "What phone numbers are in this document?",
    "Extract all monetary values",
    "Find all URLs mentioned",
    "What percentages are cited?",
    "Extract all person names",
    "Find all company names",
    "What addresses are listed?",
    "Extract all numerical data",
    "Find all product names mentioned",
    "What technical terms are used?",
    "Extract all legal references",
    "Find all medical terms",
    "What acronyms are defined?",
    "Extract all scientific citations",
    "Find all statistical values",
    "What data fields are present?",
    "Extract all quantities mentioned",
    "Find all proper nouns",
    "What measurements are reported?",
    "Extract all geographic entities",
    "Find all temporal references",
    "What categories are mentioned?",
    "Extract all financial figures",
    "Find all identifiers",
    "What entities are referenced?",
]
for q in _extraction:
    INTENT_DATA.append((q, 11))

# unknown_ambiguous (12) — 30+ examples
_unknown = [
    "Tell me something",
    "Help me",
    "What do you think?",
    "I'm curious about stuff",
    "Can you help with this thing?",
    "Hi",
    "Hello",
    "What's up?",
    "Thanks",
    "OK",
    "Yes",
    "No",
    "Maybe",
    "I don't know",
    "What?",
    "How?",
    "Why?",
    "When?",
    "Where?",
    "Who?",
    "Can you do this?",
    "Is that right?",
    "What happened?",
    "How are you?",
    "Are you sure?",
    "Interesting",
    "Tell me more",
    "Go on",
    "Okay then",
    "Let me think",
]
for q in _unknown:
    INTENT_DATA.append((q, 12))

print(f"Intent training data: {len(INTENT_DATA)} examples")


# ════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ════════════════════════════════════════════════════════════════

def train_evidence_classifier(data, output_dir, epochs=15, batch_size=16, lr=3e-5):
    """Train evidence classifier with proper hyperparameters."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    logger.info(f"Training evidence classifier: {len(data)} examples, {epochs} epochs")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    # Split into train/val (90/10)
    split = int(len(data) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == batch_labels).sum().item()
            total += len(batch_labels)

        scheduler.step()
        train_acc = correct / total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += len(batch_labels)
        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.1%} val_acc={val_acc:.1%} {'*SAVED*' if val_acc == best_val_acc else ''}")

    # Save metadata
    metadata = {
        "model": "bert-base-uncased-finetuned",
        "task": "evidence_classification",
        # HONEST DECLARATION: this legacy trainer encodes SINGLE sentences
        # (EVIDENCE_DATA holds (text, label) tuples) while the inference path
        # (neural_engine.classify_evidence) encodes (premise, hypothesis)
        # PAIRS. Artifacts trained here will FAIL the contract test
        # (tests/test_neural_contract.py) — use training/train_evidence_pairs.py,
        # the canonical pair-format trainer.
        "input_format": "single",
        "warning": "single-sentence training is incompatible with the pair-format inference path (A1 defect class); use train_evidence_pairs.py",
        "labels": ["supports", "refutes", "neutral"],
        "training_examples": len(data),
        "epochs": epochs,
        "best_val_accuracy": best_val_acc,
        "status": "fine-tuned",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Evidence classifier done. Best val accuracy: {best_val_acc:.1%}")
    return best_val_acc


def train_contradiction_detector(data, output_dir, epochs=15, batch_size=16, lr=3e-5):
    """Train contradiction detector with proper hyperparameters."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    logger.info(f"Training contradiction detector: {len(data)} examples, {epochs} epochs")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

    texts_a = [t[0] for t in data]
    texts_b = [t[1] for t in data]
    labels = [t[2] for t in data]

    split = int(len(data) * 0.9)
    train_a, val_a = texts_a[:split], texts_a[split:]
    train_b, val_b = texts_b[:split], texts_b[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_enc = tokenizer(train_a, train_b, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_a, val_b, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == batch_labels).sum().item()
            total += len(batch_labels)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += len(batch_labels)
        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.1%} val_acc={val_acc:.1%} {'*SAVED*' if val_acc == best_val_acc else ''}")

    metadata = {
        "model": "bert-base-uncased-finetuned",
        "task": "contradiction_detection",
        "input_format": "pair",
        "labels": ["contradiction", "consistent", "neutral"],
        "training_examples": len(data),
        "epochs": epochs,
        "best_val_accuracy": best_val_acc,
        "status": "fine-tuned",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Contradiction detector done. Best val accuracy: {best_val_acc:.1%}")
    return best_val_acc


def train_intent_classifier(data, output_dir, epochs=15, batch_size=16, lr=3e-5):
    """Train intent classifier with proper hyperparameters."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    num_labels = 13
    label_map = {
        0: "investigation", 1: "search", 2: "identity_analysis",
        3: "evidence_analysis", 4: "comparison", 5: "timeline",
        6: "relationship_analysis", 7: "location_analysis",
        8: "source_verification", 9: "contradiction_analysis",
        10: "summarization", 11: "extraction", 12: "unknown_ambiguous",
    }

    logger.info(f"Training intent classifier: {len(data)} examples, {epochs} epochs, {num_labels} classes")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    split = int(len(data) * 0.9)
    train_texts, val_texts = texts[:split], texts[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    train_dataset = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], torch.tensor(train_labels, dtype=torch.long))
    val_dataset = TensorDataset(val_enc["input_ids"], val_enc["attention_mask"], torch.tensor(val_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch in train_loader:
            input_ids, attention_mask, batch_labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(outputs.logits, dim=1)
            correct += (pred == batch_labels).sum().item()
            total += len(batch_labels)

        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, batch_labels = batch
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=1)
                val_correct += (pred == batch_labels).sum().item()
                val_total += len(batch_labels)
        val_acc = val_correct / val_total if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f} train_acc={train_acc:.1%} val_acc={val_acc:.1%} {'*SAVED*' if val_acc == best_val_acc else ''}")

    metadata = {
        "model": "bert-base-uncased-finetuned",
        "task": "intent_classification",
        "input_format": "single",
        "num_labels": num_labels,
        "labels": label_map,
        "training_examples": len(data),
        "epochs": epochs,
        "best_val_accuracy": best_val_acc,
        "status": "fine-tuned",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Intent classifier done. Best val accuracy: {best_val_acc:.1%}")
    return best_val_acc


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("SWEEP NEURAL ENGINE — COMPREHENSIVE RETRAINING")
    print("=" * 70)

    t0 = time.perf_counter()
    results = {}

    # Train evidence classifier
    print("\n[1/3] Training Evidence Classifier (15 epochs, 500+ examples)...")
    results["evidence"] = train_evidence_classifier(
        EVIDENCE_DATA,
        str(OUTPUT_BASE / "evidence_classifier"),
        epochs=15, batch_size=16, lr=3e-5,
    )

    # Train contradiction detector
    print("\n[2/3] Training Contradiction Detector (15 epochs, 500+ examples)...")
    results["contradiction"] = train_contradiction_detector(
        CONTRADICTION_DATA,
        str(OUTPUT_BASE / "contradiction_detector"),
        epochs=15, batch_size=16, lr=3e-5,
    )

    # Train intent classifier
    print("\n[3/3] Training Intent Classifier (15 epochs, 400+ examples)...")
    results["intent"] = train_intent_classifier(
        INTENT_DATA,
        str(OUTPUT_BASE / "intent_classifier"),
        epochs=15, batch_size=16, lr=3e-5,
    )

    elapsed = time.perf_counter() - t0

    # Quick smoke test
    print("\n" + "=" * 70)
    print("SMOKE TEST")
    print("=" * 70)

    # Force reload models
    from sweep_neural_mesh.neurons.neural_engine import _get_loader
    loader = _get_loader()
    loader._loaded = False
    loader._loading = False
    loader._failed = False
    loader.start_background_load()
    import time as _time
    _time.sleep(2)
    loader.wait_until_ready(timeout=120)

    from sweep_neural_mesh.neurons.neural_engine import NeuralEngine
    engine = NeuralEngine()
    engine.wait_until_ready(timeout=120)

    tests = [
        ("Evidence", lambda: engine.classify_evidence("Studies confirm the drug is effective", "Does the drug work?"), "supports"),
        ("Evidence", lambda: engine.classify_evidence("The drug showed no effect", "Does the drug work?"), "refutes"),
        ("Evidence", lambda: engine.classify_evidence("Results were mixed", "Does the drug work?"), "neutral"),
        ("Contradiction", lambda: engine.detect_contradiction("The drug is effective", "The drug is ineffective"), "contradiction"),
        ("Contradiction", lambda: engine.detect_contradiction("Paris is the capital", "France's capital is Paris"), "consistent"),
        ("Intent", lambda: engine.classify_intent("Investigate this person"), "investigation"),
        ("Intent", lambda: engine.classify_intent("Search for quantum computing"), "search"),
    ]

    correct = 0
    for name, fn, expected in tests:
        try:
            result = fn()
            status = "OK" if result.label == expected else "MISS"
            if result.label == expected:
                correct += 1
            print(f"  [{status}] {name}: {result.label} (conf={result.confidence:.2f}) expected={expected}")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")

    print(f"\nSmoke test: {correct}/{len(tests)} passed")

    # Save summary
    summary = {
        "training_time_seconds": elapsed,
        "results": results,
        "status": "retrained",
    }
    with open(OUTPUT_BASE / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTotal training time: {elapsed:.1f}s")
    print(f"Models saved to: {OUTPUT_BASE}")
    print("=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
