# Learning notes

Plain-English versions of every idea in this project, then the interview question each one tends to draw, and an answer that matches what this repo actually did. Read the code next to the note. The functions cited are the ones to be able to open and walk through.

## Induced vertical break

A pitch with no spin still drops, because of gravity. Induced vertical break is the extra vertical movement caused by spin, after gravity is removed. Statcast publishes it as `pfx_z`, in feet, relative to a spinless pitch that left the hand with the same velocity. This project stores it in inches: `pfx_z × 12`. Positive is ride (a four-seam hop). Negative is drop (a curveball).

On the 2024–2026 modeling frame, four-seamers average +15.7 inches and curveballs average −10.1 inches. That gap is the whole point of the feature. A four-seamer and a curve can cross the plate at the same height and have gotten there on opposite flight paths.

`api_break_z_with_gravity` is a different column. It includes the drop from gravity, so it is a large negative number on almost every pitch. Using it as IVB would tell the model that every pitch drops, which is true and useless.

**Where it lives:** `induced_vertical_break` in `src/stuffplus/physics.py`.

**Question.** Why not just use vertical movement at the plate?

**Answer.** Plate height is location. A pitcher can aim a below-average fastball at the bottom of the zone and a great fastball at the top. IVB is the spin contribution along the way, so two pitches thrown to the same spot can have different IVB, and the model is allowed to prefer one of them. Location is left out of Stuff+ on purpose. It comes back only as a diagnostic when the grade and the results disagree.

## Arm-side break and mirroring left-handers

Statcast's x axis is the catcher's right, toward first base. `pfx_x` is horizontal break in that frame. A right-hander's slider and a left-hander's slider break toward opposite ends of the field, so they have opposite signs even when they are the same pitch.

Arm-side break flips the sign for right-handers. Positive means the ball runs toward the pitcher's arm side. Negative means it sweeps toward the glove side. A right-hander's slider and a left-hander's slider then both sit on the negative side, and the model can learn "more sweep" once. The same flip is applied to horizontal release, horizontal approach angle, and the spin axis. Spin axis for a left-hander becomes `(360 − axis) mod 360`, and the model sees sine and cosine so that 359° and 1° are neighbors.

This matches Savant's own arm-side column, `api_break_x_arm`, at correlation 1.00. That check is in `metrics.json` under `arm_side_break_vs_savant_correlation`.

**Question.** Why not give the model a handedness flag and let it figure the signs out?

**Answer.** It could, and it would spend a lot of splits rediscovering the mirror. Mirroring first makes the feature mean the same thing for both hands, which also makes a SHAP chart readable: a positive push from arm-side break means run to the arm side, not "this happened to be a lefty." The test in `tests/test_physics.py` is the one to talk through. Half a foot of catcher's-right break is +6 inches for a lefty and −6 inches for a righty.

## Approach angles

Vertical approach angle (VAA) is the up-down direction of the velocity vector when the ball reaches the front of home plate. Horizontal approach angle (HAA) is the left-right direction. The front of the plate is 17 inches from Statcast's origin, so the y coordinate to solve for is `17/12` feet.

Statcast gives the acceleration fit: position and velocity at release, and a constant acceleration (`vx0, vy0, vz0, ax, ay, az`). Position in y is `y(t) = y0 + vy0 t + ½ ay t²`. Solve for the positive flight time near 0.4 seconds, then read the velocity at that time. VAA is `degrees(arctan2(vz, −vy))`. It is negative because the ball is going downhill. Four-seamers average −4.9°. Curveballs average −10.6°. A flatter fastball (closer to zero) is generally harder to hit. A steeper curve is the curve doing its job.

Savant's `attack_angle` column is not this. It is the batter's swing path. It is missing on a large share of pitches and it averages something like +9°, which is an uppercut, not a downhill pitch. It is not in the model.

HAA is mirrored the same way as horizontal break, so positive is arm side.

**Where it lives:** `time_to_plate` and `approach_angles` in `src/stuffplus/physics.py`.

**Question.** Walk me through one pitch.

**Answer.** Take the acceleration fit, solve the quadratic for the time the ball's y position equals 17/12 ft, throw away roots that are not a real flight time, then take the velocity components at that time. VAA uses vertical over the speed toward the plate. The notebook recomputes this on one 2026 four-seamer so the arithmetic is on screen. Flight times land around 0.40 to 0.48 seconds. If a root is 2 seconds or negative, the track is bad and the angle becomes missing rather than a wild feature.

## Arm angle

Statcast's `arm_angle` is the angle from the throwing shoulder to the ball. Zero is sidearm, 90 is over the top. The public pitch table does not include the shoulder coordinates, so a common proxy draws a line from a fixed sidearm height (here 2.85 ft, an old "Eckersley line") through the release point.

On this data that proxy correlates 0.72 with the measured arm angle and misses by 19° on average. That is good enough to explain the idea and not good enough to grade a pitcher. The model uses the measured column. The proxy stays in the code as a check, and the correlation is reported in the metrics so the choice is visible.

John King's 2026 sinker is a 25° arm angle. The median left-handed sinker with 100 pitches is 34°. That 9° is a real slot difference, and it would have been smeared if the model had used the proxy.

**Question.** Why did you keep a feature you don't use?

**Answer.** Because the first version of this model everyone reaches for is the proxy, and an interview will ask why you didn't. The answer is the correlation and the error, computed on the same pitches, not a preference.

## Run value

`delta_run_exp` is how much the batting team's run expectancy changed on that pitch. A home run on a 3-1 count and a home run on an 0-2 count do not move the number by the same amount, because the count had already made the at-bat more or less valuable. For a model of the pitch itself, that is noise.

The target here is count-neutral. Each pitch is replaced by the training-set average of `delta_run_exp` for its outcome bucket. Training for the 2026 grade is 2024–2025. A whiff is worth −0.114 runs. A called strike is −0.065. A foul is −0.034. A ball is +0.058. An in-play out is −0.249. A single is +0.447. A double is +0.730. A triple is +0.987. A home run is +1.527. A hit by pitch is +0.368. Those are averages, so a ball that becomes a walk is already inside the ball average, and a called strike that becomes a strikeout is already inside the called-strike average. The model never sees balls, strikes, or the base-out state.

The sign is the batter's side. Negative is good for the pitcher. That is why Stuff+ flips the scale: a lower predicted run value is a higher grade.

Outcomes that are not a real pitch result (bunts, pitchouts) are labeled `other` and dropped from the fit. The mapping is `assign_outcome` in `src/stuffplus/outcomes.py`. A unit test checks that a home run is worth more to the offense than a whiff, so a join bug cannot quietly reverse the scale.

**Question.** Why not model `delta_run_exp` directly? It is the real run value.

**Answer.** It is the real change in expectancy, and it is full of the count. The same hanging slider is a bigger number on 3-1 than on 0-2. A shape model that sees the count will learn the count. We still score the fitted model against raw `delta_run_exp` as a sanity check. On the 2026 holdout that R² is 0.014, almost the same as the 0.016 against the count-neutral target. The grade is not secretly a count model. If those two numbers had been far apart, the count-neutral target would have been the wrong simplification.

**Question.** Is a whiff really only −0.11 runs?

**Answer.** As an average that includes every count, yes, in 2024–2025. A whiff on 0-2 that ends the at-bat is worth more than a whiff on 0-0. The average is the right target for a pitch that will be thrown in every count. It is the wrong number for a leverage question, which this model does not answer.

## Stuff+

Stuff+ is a predicted run value, scaled so that 100 is average and higher is better for the pitcher.

The model is LightGBM. It sees only physical features: velocity, spin, IVB, arm-side break, release height and arm-side release, extension, measured arm angle, VAA, arm-side HAA, mirrored spin-axis sine and cosine, movement per 1,000 rpm, and the velocity and movement gaps versus the pitcher's primary fastball. It does not see pitch type, location, count, or the hitter.

The primary fastball is the most-thrown of four-seam, sinker, and cutter, with at least 50 pitches. Ties go to the four-seamer, then the sinker, then the cutter. A slider at 85 is a different pitch under a 98 mph four-seamer than under a 92 mph sinker. The gaps (`velo_diff`, `ivb_diff`, `hb_diff`) are how the model knows which of those it is looking at. On the fastball itself the gaps are about zero.

The scale, within a pitch type and season, using pitcher-seasons of at least 100 pitches:

```
Stuff+ = 100 + 10 × (league mean predicted RV − this predicted RV) / league SD
```

Ten points is one standard deviation of those qualified season means. A 110 is one standard deviation better than average for that pitch type. A 90 is one standard deviation worse. Comparing a four-seam 110 to a curveball 110 is fair. Comparing either of them to "110 across all pitches" is not, because the scale is rebuilt inside each pitch type. A sinker does not have to miss bats at a slider's rate to grade well.

Two models exist so the grade can be compared across years:

- The published 2026 grade trains on 2024 and 2025. It never saw 2026. That is the number in the app for 2026 and in the leaderboards.
- A frozen model trains on 2024 only and scores both 2025 and 2026. Year-over-year reliability uses this one. The app's 2025 grade is this one. Putting 2025 inside the training set and then quoting a 2025 grade would be grading a season the model had already memorized.

**Question.** Why is pitch-level R² only 0.016? Does that mean the model failed?

**Answer.** No. A single pitch outcome is mostly the hitter, the count, and luck. RMSE on the 2026 holdout is 0.207 runs, against a league-mean RMSE of 0.209. The boosted model beats the league mean, the pitch-type mean, and a ridge regression on the same features, but the leftover variance of one pitch is enormous. The check that matches the decision is the aggregated one: pitcher and pitch type, minimum 100 pitches, correlation 0.280 between predicted and actual run value (1,986 groups). Ridge on the same aggregation is 0.137. The pitch-type mean is 0.074. Shape beats the label, and the non-linear model beats the linear one. Inside one pitch type the label has nothing left to say, and the within-type correlations are still about 0.10 to 0.14. That is the model earning its keep.

**Question.** Why is a ridge model so close to useless here?

**Answer.** Because the good version of a pitch is an interaction. A sinker at 83 mph with −12 inches of IVB (Tyler Rogers, Stuff+ 138) and a sinker at 94 mph with +8 inches of IVB can both be good, and a linear model has to pick one direction for IVB. LightGBM can split. Ridge was kept as the baseline so that "we used boosting" is a result rather than a default. It did not win.

**Question.** How is this different from a public Stuff+?

**Answer.** The public versions differ in the target, the feature list, the seasons, and the scaling sample. Do not quote their leaderboard next to this one. What this one shares with the idea is the design: physical traits only, 100 as average, a grade that can be averaged to a pitch and compared with the results. The numbers in the README were computed here.

## The whiff model

Run value treats a called strike, a weak ground ball, and a whiff as different contributions to the same number. A coach who asks "does it miss bats?" wants the second model. It is LightGBM on the same features, trained on swings in 2024–2025, scored on 337,437 swings in 2026.

AUC 0.752. Brier 0.154. Log loss 0.476. Expected calibration error 0.004, against a 25.2% base rate. The calibration chart is a diagonal. That is the chart to show if someone asks whether the probabilities can be believed, not just ranked.

The whiff grade uses the same 100 / 10-point scale on predicted whiff rate, inside a pitch type. It correlates 0.296 with run-value Stuff+. Related, not interchangeable. King's changeup is the example: Stuff+ 93, whiff grade 98, actual whiff rate 34% against a predicted 29%. The shape is ordinary, the miss-bat rate is about average for a changeup, and the results ran hot.

**Question.** Why is sinker whiff rate so low if some sinkers grade 130?

**Answer.** League whiff rate on sinkers is 13.7% of swings. On sliders it is 33.5%. A sinker prevents runs with ground balls and called strikes. Rogers' sinker grades 138 at a 12% whiff rate because the run-value model is allowed to like weak contact. Judging it on whiff rate would call it a bad pitch. That is why both grades exist.

## Reliability

Reliability is the correlation of a measure with itself. Two versions:

**Split-half.** Draw two random samples of N pitches from the same pitcher, average the measure in each, correlate across pitchers. Repeat and average the correlations. That is how stable a sample of size N is. Done on 2026 Stuff+, whiff rate, and actual run value.

**Year-over-year.** Correlate 2025 with 2026. Stuff+ in this test is the frozen model, so neither grade was fit on its own season. ERA, FIP, and strikeout rate come from the MLB season line, at 40 innings and at 80.

| Measure | Correlation | Sample |
| --- | ---: | --- |
| Stuff+, 75 pitches, split-half | 0.71 | the first grid point at or above 0.70 |
| Stuff+, 100 pitches, split-half | 0.75 | |
| Stuff+, year over year, 200 pitches | 0.78 | 445 pitchers |
| Whiff rate, 100 swings, split-half | 0.49 | |
| Whiff rate, 300 swings, split-half | 0.72 | |
| Whiff rate, year over year, 150 swings | 0.71 | 385 pitchers |
| K%, year over year, 40 IP | 0.70 | 256 pitchers |
| FIP, year over year, 40 IP | 0.45 | 256 pitchers |
| ERA, year over year, 40 IP | 0.21 | 256 pitchers |
| Actual run value, 100 pitches, split-half | 0.07 | |
| Actual run value, 1,000 pitches, split-half | 0.32 | still under 0.70 |

Spearman-Brown is a formula that turns a correlation at one sample size into a guess at another, assuming each extra pitch adds independent information. Anchored at 100 observations, it says Stuff+ needs about 79 pitches to reach 0.70, whiff rate about 248, and actual run value about 3,100. The empirical curve is the number to quote. The formula is the assumption to be able to say out loud. For Stuff+ the two agree (75 empirical, 79 formula). For whiff rate they are close (300 empirical, 248 formula). For run value the formula is an extrapolation past the end of the curve, and it should be described that way.

**Question.** So is Stuff+ "real" at 75 pitches?

**Answer.** A 75-pitch grade correlates about 0.70 with another 75-pitch grade from the same pitcher. It is stable enough to describe the pitches in hand. It is not a season projection, and it is not a reason to change a pitcher's mix after one appearance. The leaderboards use 200 pitches. The memo pitcher has 711. At 200 pitches the split-half correlation is 0.84. That is the sample size where the grade is a trait.

**Question.** Why compare to ERA if the samples are different pitchers?

**Answer.** Because ERA is the number a front office already has, and the claim is that shape is more repeatable than ERA. The ERA sample is pitchers with 40 innings in both seasons (256 of them). The Stuff+ year-over-year sample is pitcher-seasons with 200 graded pitches both years (445). They are not the same people. The comparison is still the right one: even on a longer look, ERA repeats at 0.21 and FIP at 0.45, while Stuff+ repeats at 0.78. Strikeout rate, at 0.70, is the result that behaves most like a skill, which is what the pitching literature says and what this file reproduces.

## Clustering

Official pitch labels are a codebook two clubs do not always apply the same way. A hard slider and a soft cutter can be the same flight with different names.

The cluster model is a Gaussian mixture with diagonal covariance. It sees velocity, IVB, arm-side break, and spin. It does not see the label, the pitcher, or the grade. k runs from 8 to 12. BIC, which penalizes extra groups, picked 12 on a 60,000-pitch sample. Adjusted Rand index against the official label is 0.426. Zero would mean the clusters ignore the label. One would mean they copied it. 0.43 means they rediscovered a lot of the codebook and split the rest.

The useful split is inside a label. Cluster 1 is almost pure four-seam (93% four-seam, 94.9 mph, +16 in IVB). Cluster 5 is the riding sinker (84% sinker, 94.5 mph, +8 in IVB, +16 arm-side). Cluster 3 is the changeup family (72% changeup, 87 mph, +4 in IVB, +15 arm-side). Cluster 10 is the sweeper (83% sweeper, 83 mph, −15 arm-side). King's sinker moved from cluster 5 in 2025 to cluster 3 in 2026. The label did not change. The flight did. That is the sentence clustering is for.

**Add-a-pitch suggestions** use the fastball's cluster, and only for arsenals that already have three or fewer pitches at 5% usage. Among pitchers whose primary fastball landed in the same cluster, which pitch types do they throw that this pitcher does not, how often, and at what Stuff+? The thresholds are in `config.py`: a peer rate of at least 35% and a peer Stuff+ of at least 100, with at least 12 peer pitchers. In 2026 that rule fired 93 times. It can recommend a label the pitcher already throws under another name. The suggestion text includes the peer rate so a coach can check.

**Question.** Why a mixture model instead of k-means?

**Answer.** K-means forces every cluster to be a sphere of similar size. Pitch shapes are not. Sweepers are a tight horizontal cloud. Four-seamers are a big one. A diagonal Gaussian mixture lets each cluster have its own variance on each feature, and BIC gives a reason to stop adding clusters. K-means would have been a fine baseline. The mixture is the one that matches "these groups have different spreads."

**Question.** Why is the Rand index not higher if the clusters are any good?

**Answer.** A higher Rand index would mean the clusters were renaming FF, SI, and SL. The interesting mistakes are the point: a sinker that flies like a changeup should not be forced into the sinker group. 0.43 is high enough that the groups have names a coach recognizes, and low enough that they are not a copy of the codebook.

## SHAP

SHAP assigns each feature a share of a single prediction, in the units of the target. Here the target is run value, batter side. A negative SHAP contribution means that feature made the pitch better for the pitcher. Mean absolute SHAP is the average size of the contribution, ignoring sign, so it says which features move the grade rather than which direction they move it.

On 4,000 pitches from 2026, the order is arm-side horizontal approach angle (0.0085 runs), vertical approach angle (0.0073), arm-side break (0.0054), induced vertical break (0.0048), arm-side release (0.0045), then velocity (0.0029) and the velocity gap (0.0029). Extension and spin-axis cosine are at the bottom.

The direction still depends on the pitch. More velocity helps a four-seamer. More velocity on a curveball can mean the curve does not drop. That is why the beeswarm plot is the chart to talk through, and the mean-absolute bar is only the ranking of importance. The four-seam slice of the same sample puts VAA first (0.0093) and velocity higher than it sits in the pooled ranking (0.0038). The changeup slice puts the velocity gap in the conversation (0.0035), which is the fastball-separation feature doing what it was added to do.

**Question.** Can you trust SHAP enough to tell a pitcher which inch to change?

**Answer.** No. SHAP says which features the model used. It does not say what happens if you intervene. A pitcher who adds two inches of sweep may also lose velocity, change the spin axis, and change the arm slot. Those move together. The recommendations in this project never say "SHAP says add two inches." They say "pitchers with your hand and your velocity who already grade 110 sit here, and you are this far from that median." SHAP is the explanation of the grade. The peer table is the development target. Keeping those apart is the whole interview answer.

**Question.** Why is velocity so far down the list? Everyone knows velocity matters.

**Answer.** Velocity matters, and a lot of what velocity does shows up in the approach angle and the movement. A harder four-seamer is usually a flatter one. Once VAA and IVB are in the model, the leftover unique contribution of the mile-per-hour number is smaller. Take the angles out and velocity would shoot up the list, and the model would be worse. Importance is "what did this model still need," not "what matters in baseball."

## Season-based validation

A random 80/20 split of pitches puts the same pitcher's Tuesday start in the training set and his Sunday start in the test set. The model learns him, and the holdout looks smarter than it is.

This project trains on earlier seasons and tests on a later one.

- 2024 only, tested on 2025: LightGBM pitch-level R² 0.014, correlation 0.117, pitcher-pitch correlation 0.225. Early stopping kept 272 trees.
- 2024–2025, tested on 2026: R² 0.016, correlation 0.126, pitcher-pitch correlation 0.280. Early stopping kept 583 trees.

The 2026 model is the published grade. The 2024 model is the reliability grade. Using the published model to compare 2025 with 2026 would give 2025 an unfair advantage, because 2025 was in the fit. The app caption says which grade you are looking at.

Early stopping holds out 15% of the training rows. That slice is for tree count, not for the reported metrics. The reported metrics are the next season.

**Question.** Why not cross-validate by pitcher inside one season?

**Answer.** That is a reasonable second test, and it answers "does this generalize to a pitcher we left out?" The season split answers a harder operational question: "if we had fit this a year ago, would it have described this year?" A pitcher-grouped split inside 2026 would still train on the same run environment, the same baseball, and the same umpires. The season split does not. Both would be worth reporting. The one a front office can get burned by is the time split, so that is the one in the README.

## Calibration

A model can rank pitches correctly and still be on the wrong scale. Calibration bins the predictions and checks the average outcome in each bin.

The whiff model is calibrated. Ten bins, expected calibration error 0.004. The top bin predicted a 68.5% whiff rate and whiffed 69.3%. The bottom bin predicted 6.9% and whiffed 7.2%.

The run-value model is ranked correctly and a bit compressed. The best bin predicted +0.054 runs and came in at +0.059. The worst predicted −0.034 and came in at −0.029. Middle bins follow the line. Compression means a predicted gap between two pitches is a little small, not backwards. That is why a usage-shift estimate is described as a ceiling.

**Question.** What would you do if the reliability diagram were a curve?

**Answer.** Keep the ranking if the order of the bins is right, and do not quote the raw prediction as a run value until it is recalibrated. Platt or isotonic regression on a season that was not the test season is the usual fix. I would not fit that calibration on 2026 and then report 2026 numbers. This model was close enough on the whiff side that no second calibration was applied. The run-value compression is disclosed instead of being patched on the test set.

## Recommendations, and what they are not

`suggest_for_pitcher` only emits a note when a pre-registered comparison clears a threshold. One point of Stuff+ never becomes a recommendation.

- **Stuff ahead of results.** Arsenal Stuff+ at least 104, and results at least 0.15 runs per 100 worse than the model. In 2026: 99 pitchers. The note says command and sequencing, and it quotes fastball location scatter against the league median for that pitch. Janson Junk (Stuff+ 106, gap +0.63 runs per 100) is the Marlins example.
- **Results ahead of stuff.** Arsenal Stuff+ at most 96, and results at least 0.15 runs per 100 better than the model. In 2026: 68 pitchers. John King is this case. The note says the ERA is fragile.
- **Usage.** A pitch at Stuff+ 92 or worse, used at least 12%, at least 80 pitches, and another pitch at Stuff+ 103 or better. Shifting 10 pitches out of every 100 is converted to runs with the model's predicted run values. In 2026: 111 pitchers. King does not qualify, because nothing in the arsenal is 103 or better.
- **Shape.** A pitch under Stuff+ 98, at least 80 pitches, compared with same-handed pitchers in a ±2 mph band (widened to ±4 if fewer than eight peers) who grade 110 or better. The target is their median IVB, arm-side break, and velocity. The note states the gap in inches and refuses the causal jump. In 2026: 456 pitches got one.
- **Add a pitch.** Fastball-cluster peers throw a type this pitcher does not, at a peer rate of at least 35% and peer Stuff+ of at least 100. In 2026: 93.

**Question.** Isn't 456 shape targets just "everyone should change something"?

**Answer.** It is every pitch under Stuff+ 98, with at least 80 thrown, that has eight or more same-handed peers at 110 in the same velocity band, and that sits at least 1.5 inches away from those peers in ride or in arm-side break. A 1-inch gap does not print. The text is a measured gap, not an instruction to throw the pitch out. King's changeup gap is 4.5 inches of ride against nine peers. The function is stricter about usage, which changes a game plan, than about printing a shape comparison. The add-a-pitch rule is stricter still: it only runs when the pitcher has three or fewer pitches at 5% usage or more, so a six-pitch arsenal like King's is not told to add a seventh.

## A short script for the interview

If there is time for one minute:

"I wanted a grade of the ball itself, not the line. I trained a gradient-boosted model on velocity, spin, break, release, and approach angle, with left-handers mirrored onto the same scale, and I held out the next season. Pitch-level R² is 0.016, which is the right order of magnitude, and the grade correlates 0.28 once you average a pitcher and a pitch. It stabilizes by about 75 to 100 pitches. ERA over a full year does not. On the Marlins, John King's ERA is ahead of a 92 Stuff+, and his changeup is a slower copy of his sinker. The recommendation is a shape target taken from left-handed changeups in the same velocity band that already grade 110, not a new pitch and not a usage overhaul. I can walk through the approach-angle quadratic, the run-value table, or the reliability curve, whichever you want to pressure-test."

## Files to have open

| Idea | File |
| --- | --- |
| IVB, mirror, approach angle, arm-angle proxy | `src/stuffplus/physics.py` |
| Count-neutral run value | `src/stuffplus/outcomes.py` |
| Fastball gaps and the Stuff+ scale | `src/stuffplus/arsenal.py` |
| LightGBM, ridge, baselines, calibration | `src/stuffplus/model.py` |
| Mixture model | `src/stuffplus/cluster.py` |
| Split-half and year-over-year | `src/stuffplus/reliability.py` |
| Suggestion rules | `src/stuffplus/recommend.py` |
| How the seasons are split | `src/stuffplus/pipeline.py` |
| The numbers | `data/processed/metrics.json` |
| The memo | `reports/marlins_john_king_2026.md` |
