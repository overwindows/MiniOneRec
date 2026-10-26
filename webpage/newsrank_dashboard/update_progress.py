"""Regenerate progress.json from AML job states and upload to the static site.

Usage:
    update_progress.py [-o out.json] [--campaign JSON] [--note 'text']

Reads live AML status for the JEV / MIND job catalog in monitor_jobs.KNOWN,
under the campaign wiring below (id -> display name, metric from KNOWN).
Writes progress.json locally and uploads it to $web on
privacypshowcase12114 (published at privacypshowcase12114.z19.web.core.windows.net).

Any --campaign overrides extend/replace the builtin wiring. Notes per campaign
passed via --note become banner text.
"""
import argparse, datetime, json, os, subprocess, sys, tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "Q:/MiniOneRec/pipeline")
import monitor_jobs as M

STORE = "privacypshowcase12114"
WEB = "$web"

# campaign -> dict with display fields + run list (known_key, display_name, metric)
# motivation/settings/data/eval are static prose shown on the live page.
CAMPAIGNS = [
    {
        "name": "JEV-4B frozen-backbone readout recovery",
        "motivation": "Full-model decision SFT remains below frozen JEV. Test a small residual readout "
                      "without changing pretrained representations; prior ranking drift is observed, "
                      "but its cause is not established.",
        "settings": "Pilot source bceaab345e8e116810dc6292b7d8a189d7d1c452. Exact conditional/causal/"
                    "collated-loss parity on four real prompts. Frozen backbone; 16383 training pairs; "
                    "binary_l2_0.01 selected solely on 256 excluded training impressions. "
                    "Confirmation source a88c920ebc4c232a51df266bbb4dfc44ec477ab5; "
                    "same checksum-pinned head, no fitting or reselection. "
                    "Next train-only scale source25b9a083d93a5b3f57c0ce2781fcab3bf9ba0d13: "
                    "binary L2 0.01 fixed;131071 pairs/30944 training impressions, "
                    "same256 excluded train holdout. Exact-cache replay before reuse. "
                    "Three-arm fixed-head dev source99d4ce25c26b0fd4f172eb154d1fc86605a30165; "
                    "scaled head vs confirmed reference vs frozen from one shared feature pass. "
                    "One Standard ranking A100; six-hour timeout; no new full-model training. "
                    "Cached CPU refinement source20a8bd26abdb7ba0a84c2e2a2e651593b6f2ba43: "
                    "binary L2=0.1,0.001; pairwise L2=0.1,0.01,0.001; scaled reference wins "
                    "training-holdout ties. Four CPU threads; no new backbone extraction. "
                    "Selected binary L2=0.001,head a8bcae18921897418822c8efa3b2aa12ff8a0dbdefbc8810eb45565865f19b6c; "
                    "training-result83663c14594882d82bba3ba81402887ab5c2d8c6bebc1d793b6d63c629a28560. "
                    "Next data-only doubling sourceee17243f0d738c216b37998a780710ec7a5f10ba: "
                    "262143 pairs/69958 impressions,reuse131071 exact-prefix cache,131072 new "
                    "features. Binary L2=0.01,seed42,prompt/backbone and256 train holdout fixed; "
                    "one Standard Ranking A100/21600s,training-only,no dev scoring. "
                    "Verified doubled head2df5077c4e20ef115f6a379f23536450eefcb8a20bfc7e14534e20e49064ff17; "
                    "training-result16771883911bf4ddf695e363db8f1839f8ae35509041478e1586e434d1ee09d1. "
                    "Registered newdev comparison source8d412f4ea386f726fb355a3932a136571d7a3938; "
                    "same one-A100/six-hour cap,one frozen feature pass for all three arms. "
                    "CPU-only nonlinear binary sourceaf9eb9c0c584bfaf4638675898e7c195dba79af4: "
                    "frozen retained linear margin plus ReLU16 correction,dropout0.1,zero output init. "
                    "AdamW lr0.001/weight_decay0.01/batch2048/seed42,epochs4/8/16,"
                    "4threads/1800s. Gate:AUCgain>=0.005 and no secondary regression. "
                    "Pairwise objective control sourcede50e05c15eb9d7bf0ddebfc77142af6afa6b910: "
                    "same architecture/optimizer/checkpoints/gate,equal-impression weighted pairs. "
                    "Cross-fitted behavioral training source25b50700e63aa273e553328793c13c4d0bb81afb: "
                    "four fixed popularity features,five impression-hash folds,smoothing20;"
                    "exclude all256 holdout impressions from every count. Binary/pairwise L2=0.01. "
                    "Selected correction32e8af31786d059d90b47c32ecd118cfe0633f9c7c8d4922ed52deb703d2ff68; "
                    "populationba4d7855aabe6894c6eecd9c9a0e69f1d326200891bc174dae048f9841ecc62c; "
                    "training-result1670421674adeb8fb41f858d065edd5b276b97e3fcddb7fa0bdde23d761e63fe. "
                    "Frozen evaluation source7c6aa8b3948c7e759125570f732e74dabc148e25;"
                    "one Standard Ranking A100/21600s plus immutable CPU correction.",
        "data": "MIND_small, legacy prompt, hist30, abstracts, prompt budget4095. "
                "Pilot: dev5001-6000,1000 impressions/36802 candidates. "
                "Fixed-head confirmation: fresh dev6001-11000,5000 impressions/187881 candidates. "
                "New three-arm comparison: dev11001-16000,5000 impressions/183558 candidates. "
                "Refinement dev16001-21000:5000 impressions/186527 candidates. "
                "Next doubled-head dev21001-26000:5000 impressions/186223 candidates. "
                "Behavioral fresh-dev26001-31000:5000 impressions/188821 candidates. "
                "Popularity uses all non-holdout official training impressions,not only cached fit rows. "
                "All windows remain separate from locked first5000; do not pool intervals.",
        "eval": "Verified pilot AUC 0.63367586 versus matched frozen 0.61530684, "
                "delta +0.01836902; 95% paired CI [+0.00029635,+0.03614217] only narrowly clears zero. "
                "MRR/nDCG point values improve but their intervals include zero. "
                "All raw coverage/hashes/eight metrics/paired intervals independently rebuilt. "
                "Unchanged-head fresh5000 confirmation Completed: AUC 0.64790555 versus matched "
                "frozen 0.62637000, delta +0.02153556; 95% paired CI [+0.01380902,+0.02905105]. "
                "MRR 0.29807593->0.30990735; nDCG5 0.32653138->0.33898353; "
                "nDCG10 0.38676991->0.40229435. All four confirmation intervals exceed zero "
                "(2048 impression resamples,seed42,unadjusted); coverage,head checksum and128 "
                "feature/head spot checks independently verified. One training subset/seed, "
                "not full-dev/test or SOTA; keep pilot and confirmation intervals separate. "
                "Larger-data fit Completed: reference0.69200943->scaled0.71224853 on the reused256 "
                "TRAINING holdout only,not dev. All12 selection metrics/row exclusion/head spots "
                "independently verified; new head passes the preregistered gate. "
                "Fresh three-arm dev comparison Completed: frozen AUC0.62981627,confirmed "
                "readout0.64560843,scaled0.66443821 on the same5000 impressions. Primary "
                "scaled-minus-confirmed AUC +0.01882978,95% paired CI [+0.01445333,+0.02334820]. "
                "MRR +0.01378365,nDCG5 +0.02014485,nDCG10 +0.01783947; all four primary "
                "unadjusted intervals positive. Scaled-minus-frozen AUC +0.03462194 "
                "[+0.02724089,+0.04217320]. All12 metrics,three contrasts,raw coverage,"
                "both head checksums and128 two-head spots independently verified. No retuning. "
                "CPU refinement Completed: TRAINING-only AUC0.71224853->0.71929553 selects "
                "binary L2=0.001. MRR/nDCG10 slightly lower on the reused holdout; not an "
                "all-metric gain. All24 metrics/all6 heads' full-holdout mathematics and full "
                "training-cache digest independently verified. Frozen selected head was evaluated on "
                "previously unused dev16001-21000 against scaled reference and frozen control; "
                "primary refined-minus-scaled,no retuning. Fresh comparison Completed,source"
                "6baf350d9533f8db60c803024f914214df2c0484; one Standard Ranking A100,21600s. "
                "Same newdev5000: scaledAUC0.66751333,refined0.66938687,frozen0.62556294. "
                "Primary +0.00187354,95%CI[-0.00083710,+0.00461701]; all four primary "
                "intervals include zero. Refined head NOT promoted; retain scaled L2=0.01. "
                "Data-only doubling Completed and independently verified: reused TRAINING-only "
                "holdout AUC0.71224853->0.71305557,only+0.00080704; MRR/nDCG10 lower. "
                "All12 holdout metrics,exact262143-row sample/exclusion,both heads/provenance "
                "and128 two-head mathematical spots verified. Local controller readback failed "
                "because its20MB request truncated22,055,701-byte metadata; sample-derived "
                "bound plus extra-byte sentinel repaired verification. Same completed GPU job "
                "was NOT rerun; new2.68GB feature digest is worker-recorded,not fully downloaded. "
                "Tiny train-only AUC gain passes the registered gate,not independent dev evidence. "
                "Freeze doubled head for new21001-26000 vs retained131071 head and frozen; "
                "primary doubled-minus-retained,2048 paired resamples/seed42/unadjusted. "
                "Promote only if primary AUC95% interval positive,report secondary regressions. "
                "Actual new-dev parentkind_snake_fkcl3465qw,worker9f55bfbc-db7b-4439-9817-0303b17f64ec "
                "Completed: newdev21001-26000 AUC retained0.65561171,doubled0.65448886,"
                "frozen0.62787974. Primary doubled-minus-retained -0.00112285,"
                "95%CI[-0.00307466,+0.00088349]. All four primary point differences negative "
                "but all intervals include zero,not a statistically established regression. "
                "Doubled head NOT promoted;retain131071 head279ef6c4. Supporting retained-minus-"
                "frozen AUC +0.02773197[+0.02035031,+0.03488561]. All12 dev metrics,three "
                "contrasts,exact coverage/input hashes,both heads/training provenance and128 "
                "two-head spots independently verified atOct25 05:43 UTC+8;controller exited0. "
                "No retuning,reselection or pooling windows;no automatic further data-only scale job. "
                "Cached nonlinear binary control Completed: TRAINING-only bestepoch16 "
                "AUC0.71685153 vs retained0.71224853,only+0.00460300 below0.005;"
                "MRR0.35061941,nDCG5 0.38948949,nDCG10 0.44751655 all lower. "
                "Strict gate fails;retain exact279ef6c4,no GPU allocated. All16 training metrics,"
                "four heads' full-holdout mathematics,128spots and complete cache digest/"
                "sample/exclusion independently verified. This is a reused training holdout,"
                "not dev evidence;new GPU needs separate passed gate and dev-start>=26000. "
                "Cached nonlinear pairwise control Completed: TRAINING-only bestepoch4 "
                "AUC0.71410710 vs retained0.71224853,only+0.00185858;"
                "MRR0.35689598,nDCG5 0.39043828,nDCG10 0.45345119 all lower. "
                "Unchanged gate fails;retain exact279ef6c4. All16 pairwise metrics,four "
                "heads' full-holdout math,128spots/cache/sample/exclusion independently verified. "
                "Both CPU controllers exited0;neither nonlinear head receives a GPU test. "
                "Behavioral CPU control Completed:MIND-small TRAINING-only referenceAUC0.71224853,"
                "binary0.73529095,selected pairwise0.73622245;all secondary point metrics improve. "
                "Unchanged +0.005/all-secondary gate passes;all source integer counts,complete fit/"
                "holdout feature matrices,three full-holdout heads and12 metrics independently verified. "
                "Reused selection holdout,extra official training-log population;not dev/SOTA evidence. "
                "Freeze correction and population for newdev26001-31000;final primary hybrid-minus-retained,"
                "2048 paired resamples/seed42/unadjusted,promote only positive primary AUC95%CI. "
                "No dev fitting/reselection/count updates/pooling;report secondary regressions. "
                "Official MIND-small dev26001-31000 Completed:hybridAUC0.67411689 vs "
                "retained0.66104433,frozen0.62893530. Primary+0.01307257,"
                "95%CI[+0.00955989,+0.01672927];all four primary intervals positive. "
                "Retain hybrid32e8af31+text279ef6c4+populationba4d7855 as research candidate. "
                "Actual baseline parentkhaki_clock_p86ytlrfth,"
                "workerc33d2b74-0cea-4b9b-b5b5-8029f31dd421;bothCompleted. "
                "CPU-only recovery source60cd6e4940f0f6924a1b7dfefcfb1012795010d3 fixes repeated "
                "full-array conversion in metric aggregation;no model change or GPU rerun/budget reset. "
                "Full features/head mathematics/coverage/12metrics/three paired contrasts independently "
                "verified Oct26 01:41UTC;recovery exit0. Subset-trained on131071 pairs,not the "
                "requested full-training refit;not full-dev/test,MIND-large or SOTA. "
                "Full-training follow-up sourcee44b2e1147b5b7d637dc37d354ede943ba84ec57 "
                "uses all156965 official training impressions/469264 sampled rows,"
                "zero training holdout and no fitting-row cap. Existing seed42 negative "
                "sampling unchanged,not a5.84M-all-candidate expansion. Reuse262143 doubled "
                "cache rows and765 former-holdout rows;206356 missing features split"
                "65536/65536/65536/9748,then jointly fit every row. Fixed binary text L2=.01,"
                "pairwise behavioral L2=.01,five folds,smoothing20,four CPU threads;"
                "all3212 unpaired sampled impressions still train text/counts/normalization. "
                "Fresh official dev31001-36000/5000impressions/189095candidates compares "
                "FULL hybrid versus OLD subset hybrid,each with its own frozen correction/"
                "population. Quality pending;training loss is not a dev metric. At most four "
                "training stages plus one dev job,one Standard Ranking A100/21600s at a time,"
                "six-hour parent-creation deadlines,30h aggregate cap,no resets/duplicates. "
                "Actual full-training stage0 parentshy_celery_p9fd7cc41x submitted;"
                "server source/cache paths/stage/Ranking allocation verified. "
                "Submitted does not mean the worker is running;see live status. "
                "Initial wrong-VC queued submission was canceled.",
        "runs": [
            ("shy_celery_p9fd7cc41x", "Full-training stage0,65536 missing features;all469264 rows fit after four stages", ""),
            ("khaki_clock_p86ytlrfth", "Completed GPU baseline pass,dev26001-31000;CPU hybrid0.67411689 in Table22", "0.66104433"),
            ("c33d2b74-0cea-4b9b-b5b5-8029f31dd421", "Completed frozen-baseline worker;hybrid verified separately on CPU", ""),
            ("kind_snake_fkcl3465qw", "Doubled not promoted,newdev21001-26000 parent", "0.65448886"),
            ("9f55bfbc-db7b-4439-9817-0303b17f64ec", "Fixed doubled-head fresh5000 worker", ""),
            ("coral_turtle_b21f31930c", "Training-only data doubling parent,262143 pairs", ""),
            ("673f3dd0-32ae-4885-9ad8-f9ad0c778652", "Training-only doubled-data worker", ""),
            ("maroon_music_n00417p68k", "Refinement not promoted; newdev16001-21000 parent", "0.66938687"),
            ("907f0f26-3a75-4ca6-a2d0-ca153351814b", "Fixed refined-head fresh5000 worker", ""),
            ("khaki_muscle_yll119dfc7", "Scaled vs confirmed vs frozen fresh5000 parent", "0.66443821"),
            ("8f10cf1e-c9d4-4953-8261-ae3d0442397e", "Fixed three-arm comparison worker", ""),
            ("patient_nest_sjxf5vj85m", "Training-only readout data-scale parent", ""),
            ("6d032cc6-03de-4307-8eb3-378920c63172", "Training-only data-scale worker", ""),
            ("great_oven_88h8j3jgjl", "Fixed-head fresh5000 confirmation parent", "0.64790555"),
            ("ae73f51f-9d3c-496e-a402-8ffb37f0c5ed", "Fixed-head confirmation worker", ""),
            ("ivory_star_h4z4j4tn84", "Frozen-backbone readout pilot dev5001-6000", "0.63367586"),
            ("240ffc68-83e6-4289-985b-48ab9b966045", "One-A100 readout pilot worker", ""),
            ("teal_hamster_1nnbmw8xy9", "Canceled initial wrong-VC parent", ""),
            ("4778df36-1bc5-42b6-b61f-418d51d448c3", "Initial wrong-VC worker; cancellation requested", ""),
        ],
    },
    {
        "name": "JEV-4B Campaign-1: SFT (MIND small → large)",
        "motivation": "Stock OpenJev is a frozen 'no-UL' ranking model (0.6432 AUC) that never saw MIND news. "
                      "Fine-tune it to P(yes) on MIND pointwise samples to build a news-specialized ranking "
                      "model — the foundation the RLCD / RL / Teacher-OPD campaigns build on.",
        "settings": "Qwen3.5 stacked linear-attn JEV-4B · noul A/B prompt · plain pointwise SFT CE on the "
                    "answer token (R-Drop self-KL collapsed to chance 0.4975 and was removed) · sdpa · "
                    "micro-batch 2 · lr 1e-4 · DS ZeRO-2 · abstract, hist30, neg1.0, cutoff4096",
        "data": "MIND_small first (fast validation loop) → re-scale to MINDlarge once signal is confirmed",
        "eval": "run_jev_eval.py noul scorer on dev@5000 → recompute pointwise AUC vs stock JEV 0.6432",
        "runs": [
            ("sincere_garlic_7t99m7nxw6", "C1fix SFT CE (MIND_small) train", ""),
            ("boring_car_qdw46wzq1k", "C1fix SFT CE (MIND_small) dev@5000 eval", "0.4959"),
        ],
    },
    {
        "name": "JEV-4B matched frozen prompt controls",
        "motivation": "Separate prompt effects from weight adaptation before evaluating the candidate-only loss.",
        "settings": "Same frozen weights, hist30, abstracts, 4095 prompt-token budget, one ranking Standard A100; "
                    "immutable source 10b4379407d11e3e3a00c6cae3a75042eeaf731c",
        "data": "Same first 5000 MIND_small dev impressions / 187009 candidates; data hashes and raw labels checked",
        "eval": "Legacy AUC 0.63623517; native-click AUC 0.63313036. Paired AUC delta -0.00310481, "
                "95% impression-bootstrap interval [-0.00711694, +0.00112362] includes zero "
                "(2048 resamples, seed42; no multiplicity adjustment). No prompt-only gain established. "
                "All four metrics independently reconstructed from raw scores.",
        "runs": [
            ("bright_ear_vnr75xb8tw", "Frozen legacy prompt dev@5000", "0.63623517"),
            ("bright_ear_vnr75xb8tw", "Frozen native-click prompt dev@5000", "0.63313036"),
        ],
    },
    {
        "name": "JEV-4B full-data candidate-only loss ablation",
        "motivation": "Isolate the loss change after frozen controls found no native-prompt AUC gain.",
        "settings": "decision BCE(z_A-z_B) + legacy prompt; no KL; seed42; lr1e-4; "
                    "global batch128/micro2; three epochs; bf16/ZeRO-2; one ranking Standard "
                    "eight-A100 node; 72-hour command timeout; immutable source ec86d37eb4b6",
        "data": "All 469264 deterministic sampled MIND_small training pairs; hist30, abstracts, "
                "cutoff4096; full audit passed without truncation",
        "eval": "Training Completed: all 11001 steps/3 epochs, 44.47 training hours; final checkpoint "
                "metadata/tokenizer/427-tensor extent verified. Fresh matched old full-vocabulary "
                "AUC 0.54519484 independently reconstructed with full raw-score coverage and input hashes. "
                "New decision AUC 0.56365540: +0.01846056 versus old SFT "
                "(95% paired CI [+0.00735828,+0.02897442]), but -0.07257978 versus matched frozen "
                "(CI [-0.08375560,-0.06140528]). All 5000/187009 coverage, hashes, four metrics "
                "and paired intervals independently verified; one training seed, no multiplicity adjustment. "
                "Partial recovery, not successful adaptation or SOTA; downstream campaigns remain gated.",
        "runs": [
            ("mighty_milk_0znks9vwg7", "Full decision-loss + legacy-prompt training", ""),
            ("2c74fe4d-5b03-4df1-b2a3-502003a390eb", "Full decision-loss training worker", ""),
            ("purple_flag_yjyhnhtpyn", "Old full-vocabulary matched legacy dev@5000", "0.54519484"),
            ("helpful_lizard_77ghb5kb5b", "New decision-loss matched legacy dev@5000", "0.56365540"),
        ],
    },
    {
        "name": "JEV-4B Campaign-2: RLCD",
        "motivation": "Distill click preferences without an online reward model: score candidates with the "
                      "Campaign-1 model, build contrastive prefs, fine-tune to raise P(A) on preferred vs "
                      "dispreferred (DPO-style).",
        "settings": "DPO-style contrastive distillation from SFT-JEV self-scores · noul A/B contract",
        "data": "MIND_small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "JEV-4B Campaign-3: RL (GRPO)",
        "motivation": "Reinforcement learning from a pointwise reward over noul A/B responses on the "
                      "Campaign-1 checkpoint, reusing the existing verl/GRPO + vLLM infra.",
        "settings": "verl GRPO · reward_type=pointwise_binary / mind_auc · vLLM rollout · on C1 checkpoint",
        "data": "MIND small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "JEV-4B Campaign-4: Teacher OPD",
        "motivation": "Offline preference distillation from a strong external teacher (our L1.3 1.7B best "
                      "pointwise ranker) into JEV-4B — a real, distinct teacher signal (unlike the failed "
                      "within-model R-Drop).",
        "settings": "KL-to-frozen-teacher logits term on JEV noul format · β≈0.1",
        "data": "MIND small smoke → MINDlarge",
        "eval": "dev@5000 pointwise AUC",
        "runs": [],
    },
    {
        "name": "Qwen3.5-2B reference pointwise (r11b/r12)",
        "motivation": "Same linear-attention family as JEV; reference pointwise ranker for comparison.",
        "settings": "abstract gckpt · r12: FULL MIND_small no sample cap (EP5 ckpt4096) · r11b: MINDlarge s300k (30万样本上限)",
        "data": "r12: MIND_small (no cap) · r11b: MINDlarge (s300k truncated)",
        "eval": "dev@5000 AUC",
        "runs": [
            ("neat_band_393lt71y7t", "r7 chat abstract gckpt", ""),
            ("happy_energy_zrvmghx41r", "r7 Base abstract gckpt", ""),
            ("serene_rice_88lx58jm5j", "r11b chat SFT (MINDlarge s300k) train", ""),
            ("olive_steelpan_spqkjfn91t", "r11b chat dev@5000 eval", "0.6542"),
            ("magenta_pasta_7j0yzmwxhw", "r11b Base SFT (MINDlarge s300k) train", ""),
            ("frosty_whistle_tlztv5jz9h", "r11b Base dev@5000 eval", "0.6272"),
            ("zen_sun_11q5r489j4", "r11b chat FULL-DEV eval (376k)", ""),
            ("nice_plate_8v2hbbvb5y", "r11b Base FULL-DEV eval (376k)", ""),
            ("green_salt_0m0w4lhcgz", "r12 chat FULL MIND_small (no cap) train", ""),
            ("tender_box_q646hppnl4", "r12 Base FULL MIND_small (no cap) train", ""),
        ],
    },
    {
        "name": "Qwen3.5-2B P-series rerun (MIND_small L1.x sweep)",
        "motivation": "Rerun the historical Qwen3-1.7B MIND_small P1.x sweep on Qwen3.5-2B — identical settings, only model changes.",
        "settings": "bs256 micro4 · train_sample=0 (full MIND_small) · chat template · gckpt · cutoff8192 (P1.3=4096)",
        "data": "MIND_small",
        "eval": "dev@5000 AUC",
        "runs": [
            ("affable_hamster_dfksgqqfjm", "2B P1.0 ep5 base (small)", "0.6360"),
            ("dreamy_monkey_rb7j50sqc3", "2B P1.1 neg3.0 (small)", "0.6287"),
            ("olden_fennel_zvl1kjpwnz", "2B P1.2 ep7 (small)", "0.6424"),
            ("keen_drain_9d0g467ccp", "2B P1.3 abstract ckpt4096 (small)", "0.6385"),
            ("serene_station_c7g9br7ph1", "2B P1.4 hist50 (small)", "0.6429"),
            ("neat_potato_7myz9f6k4w", "2B P1.5 neg3 ep7 hist50 (small)", "0.6189"),
            ("gentle_root_d1c59cx392", "2B P1.7 subcategory (small)", "0.6253"),
        ],
    },
    {
        "name": "Qwen3.5-2B L-series rerun (MIND_small — research phase)",
        "motivation": "Rerun the historical Qwen3-1.7B L1.x sweep (incl. F1.1/H1.1) on Qwen3.5-2B — identical settings, only model changes. RESEARCH FIRST on MIND_small; winners promote to MIND_large.",
        "settings": "bs256 micro4 · train_sample=0 (full MIND_small) · chat template (L1.7/L1.9 Base no-chat) · gckpt · cutoff8192",
        "data": "MIND_small",
        "eval": "dev@5000 AUC",
        "runs": [
            ("polite_vulture_6qxy937lfx", "2B L1.1 neg3.0 (small research)", "0.6287"),
            ("sleepy_cassava_4qtqckylh0", "2B L1.2 ep7 (small research)", "0.6424"),
            ("gentle_soca_k0yplhlqdz", "2B L1.3 abstract (small research)", ""),
            ("placid_nut_mvf48tgbm3", "2B L1.4 hist50 (small research)", "0.6429"),
            ("lemon_pasta_t8ct4b0nkx", "2B L1.5 neg3 ep7 hist50 (small research)", "0.6189"),
            ("cool_bottle_9hwv3v9tl4", "2B-Base L1.7 (small, no chat)", ""),
            ("nice_peach_wzg73q6dl8", "2B-Base L1.9 abstract (small research)", ""),
            ("sweet_cassava_33crdfhp1d", "2B L1.13 abstract ep7 neg3 (small research)", ""),
            ("blue_plum_21l227nncx", "2B F1.1 abstract+subcat (small research)", ""),
            ("salmon_squash_t0b7yt5zls", "2B H1.1 hard-neg 100% (small) rerun hardneg-fix", ""),
            ("goofy_box_nghcbmkp1z", "2B H1.1 (OLD — collision w/ P1.0, unreliable)", ""),
        ],
    },
]

HEADLINE = (
    "Verified official MIND-small dev26001-31000:hybridAUC0.67411689 vsretained0.66104433,"
    "delta+0.01307257,95%CI[+0.00955989,+0.01672927];all four primary intervals positive. "
    "New research candidate is the hybrid;subset-trained on131071 pairs,NOT full-training/SOTA. "
    "CPU-only recovery complete,no GPU rerun;all12 metrics/three contrasts/full features verified. "
    "MIND-small behavioral TRAINING-only gate passed:AUC0.71224853->0.73622245,"
    "all secondary point metrics improve;not dev,test/leaderboard or SOTA. "
    "Official-dev result covers5000 impressions/188821candidates. Full-training follow-up uses "
    "all156965 official impressions/469264 sampled rows,zero held-out training impressions,"
    "no fitting-row cap;quality pending untouched official dev31001-36000/189095candidates. "
    "Both cached nonlinear controls rejected before GPU:binary bestTRAINING-AUC0.71685153,"
    "pairwise0.71410710 vs retained0.71224853;both below+0.005 gate and both lowerMRR/nDCG. "
    "Retain exact279ef6c4;all32 training metric cells/full-holdout head math verified,no newdev claim. "
    "Doubled head NOT promoted: untouched dev21001-26000 retainedAUC0.65561171 vs doubled0.65448886 "
    "(frozen0.62787974). Primary -0.00112285,95%CI[-0.00307466,+0.00088349];"
    "all four primary intervals include zero. Retain131071 head279ef6c4. "
    "All12 dev metrics,three contrasts,both heads and128spots independently verified;both jobsCompleted,"
    "controller exited0. No retuning,pooling or automatic further data-only scaling. "
    "Doubling Completed:training-only AUC0.71224853->0.71305557,only+0.00080704;MRR/nDCG10 lower. "
    "262143 pairs,reuse131071 cached,131072 new features;all12 holdout metrics/128spots verified. "
    "Metadata-only readback repair,no GPU rerun. Frozen doubled head2df5077c vs retained279ef6c4 "
    "and frozen on untouched dev21001-26000/186223candidates;fixed comparison Completed,"
    "one A100/six-hour cap;train-only gain did not establish a dev improvement. "
    "Refinement NOT promoted: primary newdev AUC +0.00187354 "
    "(95%CI[-0.00083710,+0.00461701]); all four primary intervals include zero. "
    "Retain scaled L2=0.01 head; no dev retuning or replacement selection. "
    "CPU refinement Completed: training-only AUC0.71224853->0.71929553 selects binary L2=0.001; "
    "not dev,and MRR/nDCG10 slightly lower. Selected head frozen for newdev16001-21000 "
    "against scaled reference and frozen; fresh5000 comparison Completed,one A100/six hours. "
    "Same window: scaled0.66751333,refined0.66938687,frozen0.62556294. "
    "Larger-data fit Completed: training-only holdout AUC0.69200943->0.71224853,selection only,not dev. "
    "Three-arm fresh dev11001-16000 Completed: frozen AUC0.62981627,confirmed0.64560843,"
    "scaled0.66443821. Primary scaled-minus-confirmed +0.01882978 "
    "(95% paired CI [+0.01445333,+0.02334820]); all four primary metric intervals positive. "
    "No dev fitting or retuning; all12 metrics,three contrasts,both heads and128 spots verified. "
    "Frozen-readout gain confirmed on fresh dev6001-11000: matched AUC 0.62637000 -> 0.64790555, "
    "+0.02153556 (95% paired CI [+0.01380902,+0.02905105]); MRR/nDCG also improve with "
    "positive unadjusted intervals. Exact unchanged head, no retraining/retuning; all5000 "
    "impressions/187881 candidates and128 feature/head spot checks independently verified. "
    "Earlier dev5001-6000 pilot AUC 0.61530684 -> 0.63367586 remains a separate comparison. "
    "Both jobs Completed; one training subset/seed, not full-dev/test validation or SOTA. "
    "Recorded competition target: 0.7326; best submitted ensemble: 0.7144. "
    "Matched frozen controls completed 2026-10-20: legacy AUC 0.6362 vs native-click 0.6331; "
    "paired AUC interval includes zero, so no prompt-only improvement is established. "
    "Full input audit passed all 469264 training pairs and 187009 dev candidates without truncation. "
    "Full-data decision-loss + legacy-prompt experiment Completed and independently verified: "
    "decision AUC 0.56365540 versus old SFT 0.54519484 and matched frozen 0.63623517. "
    "Decision-minus-old +0.01846056 (95% paired CI [+0.00735828,+0.02897442]); "
    "still below frozen by 0.07257978. Partial recovery, not successful adaptation or SOTA. "
    "One training seed; impression-bootstrap intervals unadjusted; downstream campaigns remain gated."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress.json"))
    ap.add_argument("--note", action="append", default=[], help="campaign:note")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    # merge --note into a {campaign: note} map
    note_map = {}
    for ent in args.note:
        if ":" in ent:
            k, v = ent.split(":", 1)
            note_map[k] = v

    ml = M.client()
    campaigns = []
    for camp in CAMPAIGNS:
        cname, runs = camp["name"], camp["runs"]
        cl = []
        for key, disp, metric in runs:
            status = None
            try:
                j = ml.jobs.get(key)
                status = str(j.status).replace("JobStatus.", "")
            except Exception:
                status = "unknown"
            cl.append({"id": key, "name": disp, "metric": metric or "", "status": status})
        campaigns.append({
            "name": cname,
            "runs": cl,
            "motivation": camp.get("motivation", ""),
            "settings": camp.get("settings", ""),
            "data": camp.get("data", ""),
            "eval": camp.get("eval", ""),
            "notes": note_map.get(cname, ""),
        })

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = {"generated": generated, "headline": HEADLINE, "campaigns": campaigns}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    if not args.no_upload:
        cmd = (f"az storage blob upload --account-name {STORE} "
               f"--container-name {WEB} --file {args.out} --name progress.json --overwrite")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            print("UPLOAD FAILED:", r.stderr[-800:])
        else:
            print("uploaded progress.json ->", gen_url(), flush=True)

    print("generated", args.out)


def gen_url():
    return f"https://{STORE}.z19.web.core.windows.net/progress.json"


if __name__ == "__main__":
    main()
