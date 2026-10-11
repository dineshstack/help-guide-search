# Answer agent: evaluation report

Stage 1 found that search never refuses: it returns its three closest guides even for a question the guides do not cover. This report measures the answer agent, which reads the passages search found: how often it answers from an expected guide, and how often it declines the questions the guides cannot answer. The numbers are reported as measured.

## What was run

- Questions: 50 reviewed and scored; 10 not reviewed and not scored (the Arabic ones).
- Models: `claude-fable-5-1`, `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`; effort: medium; every other request setting at its default.
- Each question was asked once per model. Run dates: 2026-10-11.
- Search: the default mode with the committed threshold. The agent sees up to 3 guides with up to 2 passages each.
- Instructions: the version with hash `916a906d0da7`. They were adjusted on `dev` only.

## How to read the tables

- **Answered, citing an expected guide**: the agent answered and at least one guide it cited is one a reviewer listed for the question.
- **Answered, citing only other guides**: the agent answered from guides the reviewer did not list. The answer may still be right; nobody judged its text.
- **Handed over or asked back**: the agent declined a question the guides can answer.
- **Unanswerable: declined**: the correct outcome for a question no guide covers. **Unanswerable: answered** is the wrong one.
- **Needed a rewrite**: the first draft failed the check and the model was asked once more.
- **Cost** is estimated from the token counts and the public prices of 11 October 2026.

## Results on test

The `test` questions played no part in writing the instructions.

| Model | Answered, citing an expected guide | Answered, citing only other guides | Handed over or asked back | Unanswerable: declined | Unanswerable: answered | Needed a rewrite | Tokens per question (in / out) | Cost per 100 questions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-fable-5-1` | 13 of 20 (65%) | 0 of 20 (0%) | 7 of 20 (35%) | 5 of 5 (100%) | 0 of 5 (0%) | 0 of 25 (0%) | 1902 / 175 | $2.78 |
| `claude-opus-5-5` | 13 of 20 (65%) | 0 of 20 (0%) | 7 of 20 (35%) | 5 of 5 (100%) | 0 of 5 (0%) | 0 of 25 (0%) | 1902 / 180 | $1.12 |
| `claude-sonnet-5-5` | 13 of 20 (65%) | 0 of 20 (0%) | 7 of 20 (35%) | 5 of 5 (100%) | 0 of 5 (0%) | 0 of 25 (0%) | 1902 / 119 | $0.50 |
| `claude-haiku-5-5` | 14 of 20 (70%) | 0 of 20 (0%) | 6 of 20 (30%) | 5 of 5 (100%) | 0 of 5 (0%) | 0 of 25 (0%) | 1902 / 257 | $0.03 |

### Results on test that are not what one would want

#### `claude-fable-5-1`

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-025` After the insurer agreed the repair we stripped the car and found more damage. How do I claim the extra?
  - expected: `manage-insurance-claims`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing

#### `claude-opus-5-5`

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-025` After the insurer agreed the repair we stripped the car and found more damage. How do I claim the extra?
  - expected: `manage-insurance-claims`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing

#### `claude-sonnet-5-5`

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-025` After the insurer agreed the repair we stripped the car and found more damage. How do I claim the extra?
  - expected: `manage-insurance-claims`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing

#### `claude-haiku-5-5`

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing

## Results on dev

The instructions were adjusted while looking at these questions.

| Model | Answered, citing an expected guide | Answered, citing only other guides | Handed over or asked back | Unanswerable: declined | Unanswerable: answered | Needed a rewrite | Tokens per question (in / out) | Cost per 100 questions |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `claude-fable-5-1` | 15 of 20 (75%) | 0 of 20 (0%) | 5 of 20 (25%) | 3 of 5 (60%) | 2 of 5 (40%) | 0 of 25 (0%) | 1881 / 249 | $3.13 |
| `claude-opus-5-5` | 15 of 20 (75%) | 1 of 20 (5%) | 4 of 20 (20%) | 3 of 5 (60%) | 2 of 5 (40%) | 0 of 25 (0%) | 1881 / 253 | $1.26 |
| `claude-sonnet-5-5` | 15 of 20 (75%) | 0 of 20 (0%) | 5 of 20 (25%) | 4 of 5 (80%) | 1 of 5 (20%) | 1 of 25 (4%) | 1968 / 170 | $0.56 |
| `claude-haiku-5-5` | 15 of 20 (75%) | 0 of 20 (0%) | 5 of 20 (25%) | 3 of 5 (60%) | 2 of 5 (40%) | 1 of 25 (4%) | 1983 / 400 | $0.04 |

### Results on dev that are not what one would want

#### `claude-fable-5-1`

- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-016` Storekeeper refused my parts request by mistake, can he undo it?
  - expected: `request-and-issue-parts`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide; the agent: answer, citing `start-using-sam`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide; the agent: answer, citing `record-invoice-payments`, `frequently-asked-questions`

#### `claude-opus-5-5`

- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`; the agent: answer, citing `manage-service-bookings`
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-016` Storekeeper refused my parts request by mistake, can he undo it?
  - expected: `request-and-issue-parts`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide; the agent: answer, citing `start-using-sam`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide; the agent: answer, citing `record-invoice-payments`, `frequently-asked-questions`

#### `claude-sonnet-5-5`

- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-016` Storekeeper refused my parts request by mistake, can he undo it?
  - expected: `request-and-issue-parts`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide; the agent: answer, citing `start-using-sam`

#### `claude-haiku-5-5`

- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-016` Storekeeper refused my parts request by mistake, can he undo it?
  - expected: `request-and-issue-parts`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`; the agent: hand_over (the_guides_do_not_answer_this), citing nothing
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide; the agent: answer, citing `start-using-sam`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide; the agent: answer, citing `record-invoice-payments`, `frequently-asked-questions`

## Limits

- Citing an expected guide is not the same as answering correctly. Judging the text of an answer needs a person or another model, and was not done.
- The set is small: about 20 answerable and 5 unanswerable questions per split. One unanswerable question is 20 points.
- Each question was asked once. A model's output varies from run to run, so a second run could move a result by a question or two.
- The agent only sees what search returned. When search misses the right guide, the best the agent can do is decline.
- The expected guides are one reviewer's labels.
- Only English questions are scored. The Arabic ones wait for a reviewer who reads Arabic.
