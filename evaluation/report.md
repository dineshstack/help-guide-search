# Evaluation report

This report answers one question: is search better than letting a model pick guides from a list of their titles? The numbers are reported as measured, whichever way they come out.

## What was run

- Guides: the snapshot fetched on 2026-10-10, 60 guides, 335 passages in the index
- Questions: the question file at commit `a5a66f02768827ecdbc6356aa71a903a76aaaa98`; 50 reviewed and scored (25 dev, 25 test); 10 not reviewed and not scored
- Embedding model: `intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Title-picking model: `claude-sonnet-4-6`; each question was asked once; answers recorded on 2026-10-11; its prompt was not revised after the first run
- Threshold: 0.0000. Below this best similarity, vector and hybrid search return no match.
- Test questions: scored in this run. No results file from an earlier run was found in the output folder.
- Run date: 2026-10-11
- Machine: AMD EPYC 9354P 32-Core Processor, 8 cores, 31 GB memory, Linux, Python 3.13.16; this container is limited to 2 CPUs and 3 GB memory

**This report covers English only.** No Arabic question has been reviewed yet, so none is scored. When a reader of Arabic has reviewed them, the threshold is chosen again on `dev` and this report is rebuilt.

## Headline: hybrid search against title-picking

| Measure | Hybrid | Title-picking | Hybrid right, title-picking wrong | Title-picking right, hybrid wrong | p (two-sided sign test) |
| --- | --- | --- | --- | --- | --- |
| hit@1 | 10 of 20 (50%) | 17 of 20 (85%) | 2 | 9 | 0.065 |
| hit@3 | 14 of 20 (70%) | 20 of 20 (100%) | 0 | 6 | 0.031 |

- hit@1: **no difference that this set can show** between hybrid search and title-picking.
- hit@3: hybrid search is **worse** than title-picking.

This is the comparison that was named before any method was run: the English `test` questions the guides can answer, on hit@1 and on hit@3. The two counts are the questions that only one of the two methods got right. Questions that both got right, or both got wrong, do not enter the test. "Better" or "worse" is said only when an exact two-sided sign test on the two counts gives p below 0.05. All other numbers in this report are descriptive.

## How to read the tables

- **hit@1**: the first guide returned is an expected one.
- **hit@3**: an expected guide is among those returned.
- **Passage hit@3**: hit@3, and one of that guide's returned passages comes from a section that holds the answer. For title-picking it equals hit@3, because the whole guide is handed over.
- **MRR**: 1 divided by the position of the first expected guide (1, 1/2, 1/3, or 0 if it is absent), averaged over the questions.
- **Wrong refusals**: questions the guides can answer for which nothing was returned. Such a question scores 0 on every measure above.
- **Guides returned**: the average number of guides returned per question that was not refused. A method may return fewer than three.
- **Text handed over**: characters, as the median over the questions that were not refused. For search it is the total length of the returned passage texts. For title-picking it is the total length of the picked guides' whole bodies. Titles, headings and URLs are not counted.
- **Correct refusals**: questions the guides cannot answer for which nothing was returned. The rest were answered when they should have been refused.

Every rate is shown with its count. `dev` and `test` are never added together, because the threshold was fitted on `dev`.

## Results on test

The `test` questions played no part in any choice. The threshold was frozen before they were scored.

### English questions the guides can answer

| Method | hit@1 | hit@3 | Passage hit@3 | MRR | Wrong refusals | Guides returned | Text handed over (characters) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Title-picking | 17 of 20 (85%) | 20 of 20 (100%) | 20 of 20 (100%) | 0.925 | 0 of 20 (0%) | 2.9 | 6,864 |
| Vector | 8 of 20 (40%) | 13 of 20 (65%) | 13 of 20 (65%) | 0.500 | 0 of 20 (0%) | 3.0 | 2,195 |
| Keyword | 4 of 20 (20%) | 9 of 20 (45%) | 6 of 20 (30%) | 0.292 | 0 of 20 (0%) | 3.0 | 2,784 |
| Hybrid | 10 of 20 (50%) | 14 of 20 (70%) | 12 of 20 (60%) | 0.575 | 0 of 20 (0%) | 3.0 | 2,632 |

### Arabic questions the guides can answer

No reviewed question in this group.

### Questions the guides cannot answer

| Method | Correct refusals | Answered when it should have refused |
| --- | --- | --- |
| Title-picking | 0 of 5 (0%) | 5 of 5 (100%) |
| Vector | 0 of 5 (0%) | 5 of 5 (100%) |
| Keyword | 0 of 5 (0%) | 5 of 5 (100%) |
| Hybrid | 0 of 5 (0%) | 5 of 5 (100%) |

### Ranking only: the threshold switched off

Vector and hybrid search, run again with the threshold at 0, so that nothing is refused. When a result above is weak, this shows whether the ranking or the refusing is the cause.

| Method | Questions | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- | --- |
| Vector | English | 8 of 20 (40%) | 13 of 20 (65%) | 0.500 |
| Hybrid | English | 10 of 20 (50%) | 14 of 20 (70%) | 0.575 |

## Results on dev

The threshold was chosen on these questions. The refusal numbers of vector and hybrid search are therefore better here than they would be on new questions.

### English questions the guides can answer

| Method | hit@1 | hit@3 | Passage hit@3 | MRR | Wrong refusals | Guides returned | Text handed over (characters) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Title-picking | 18 of 20 (90%) | 20 of 20 (100%) | 20 of 20 (100%) | 0.950 | 0 of 20 (0%) | 2.6 | 6,776 |
| Vector | 6 of 20 (30%) | 12 of 20 (60%) | 9 of 20 (45%) | 0.417 | 0 of 20 (0%) | 3.0 | 2,325 |
| Keyword | 7 of 20 (35%) | 12 of 20 (60%) | 10 of 20 (50%) | 0.475 | 0 of 20 (0%) | 3.0 | 2,726 |
| Hybrid | 11 of 20 (55%) | 16 of 20 (80%) | 14 of 20 (70%) | 0.650 | 0 of 20 (0%) | 3.0 | 2,723 |

### Arabic questions the guides can answer

No reviewed question in this group.

### Questions the guides cannot answer

| Method | Correct refusals | Answered when it should have refused |
| --- | --- | --- |
| Title-picking | 0 of 5 (0%) | 5 of 5 (100%) |
| Vector | 0 of 5 (0%) | 5 of 5 (100%) |
| Keyword | 0 of 5 (0%) | 5 of 5 (100%) |
| Hybrid | 0 of 5 (0%) | 5 of 5 (100%) |

### Ranking only: the threshold switched off

Vector and hybrid search, run again with the threshold at 0, so that nothing is refused. When a result above is weak, this shows whether the ranking or the refusing is the cause.

| Method | Questions | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- | --- |
| Vector | English | 6 of 20 (30%) | 12 of 20 (60%) | 0.417 |
| Hybrid | English | 11 of 20 (55%) | 16 of 20 (80%) | 0.650 |

## Time

| Method | Questions | Median | Maximum |
| --- | --- | --- | --- |
| Title-picking | 50 | 1,582 ms | 7,686 ms |
| Vector | 50 | 34 ms | 43 ms |
| Keyword | 50 | 3 ms | 8 ms |
| Hybrid | 50 | 35 ms | 48 ms |

Per question, over all scored questions together. Search is timed on the machine named above, around the search function, including embedding the question, after one discarded warm-up query. Title-picking time is the model's response time over the network, recorded when the baseline was run. These are different kinds of time. They are shown side by side and are not compared.

## The threshold

Vector and hybrid search return no match when the best similarity of a question is below the threshold. It is chosen on the reviewed `dev` questions only. The candidates are 0, which refuses nothing, and the midpoint between each pair of neighbouring best similarities. A candidate handles a question correctly when it lets through a question the guides can answer, or refuses one they cannot. The candidate with the highest count is used, and the lowest of them on a tie. The same value serves vector and hybrid search, and both languages.

Threshold in `settings.toml`, used in this run: 0.0000.
The rule, applied to this run's dev scores, gives the same value.

| Candidate | Dev questions handled correctly |  |
| --- | --- | --- |
| 0.0000 | 20 of 25 | chosen |
| 0.8090 | 19 of 25 |  |
| 0.8159 | 18 of 25 |  |
| 0.8185 | 19 of 25 |  |
| 0.8212 | 18 of 25 |  |
| 0.8242 | 17 of 25 |  |
| 0.8322 | 18 of 25 |  |
| 0.8392 | 17 of 25 |  |
| 0.8406 | 16 of 25 |  |
| 0.8445 | 15 of 25 |  |
| 0.8490 | 16 of 25 |  |
| 0.8505 | 15 of 25 |  |
| 0.8517 | 16 of 25 |  |
| 0.8532 | 15 of 25 |  |
| 0.8546 | 16 of 25 |  |
| 0.8556 | 15 of 25 |  |
| 0.8566 | 14 of 25 |  |
| 0.8603 | 13 of 25 |  |
| 0.8635 | 12 of 25 |  |
| 0.8643 | 11 of 25 |  |
| 0.8653 | 10 of 25 |  |
| 0.8665 | 9 of 25 |  |
| 0.8710 | 8 of 25 |  |
| 0.8750 | 7 of 25 |  |
| 0.8767 | 6 of 25 |  |

The rule's choice is 0.0000.

- There is no dev score below it.
- Nearest dev score at or above it: 0.8026 away

The best similarity of the dev questions, by group:

| Dev questions | How many | Lowest | Median | Highest |
| --- | --- | --- | --- | --- |
| English, answerable | 20 | 0.8026 | 0.8556 | 0.8784 |
| Arabic, answerable | 0 | - | - | - |
| Unanswerable | 5 | 0.8165 | 0.8481 | 0.8539 |

The rule chose 0. On these questions similarity did not separate the questions the guides cannot answer from the ones they can. With a threshold of 0, vector and hybrid search never refuse.

## Misses

A miss is a question the guides can answer for which no expected guide was among those returned, or a question the guides cannot answer that was answered anyway.

### Misses on test

#### Title-picking

- `na-001` Can I link this to QuickBooks so my accountant gets our bills without retyping them?
  - expected: no guide (the guides do not cover this)
  - returned: `frequently-asked-questions`, `overview`, `financial-reports-and-statements`
- `na-003` The same customer got saved twice under two phone numbers. How do I join the two together?
  - expected: no guide (the guides do not cover this)
  - returned: `create-and-manage-customers`, `frequently-asked-questions`
- `na-004` I'm opening a second branch next year. How many branches can one account handle?
  - expected: no guide (the guides do not cover this)
  - returned: `frequently-asked-questions`, `overview`, `plans-trials-and-billing`
- `na-006` Can the customer sign on a tablet when he drops off the car?
  - expected: no guide (the guides do not cover this)
  - returned: `maintenance-and-driver-handovers`, `inspect-a-vehicle`, `create-and-manage-work-orders`
- `na-010` Can we scan part barcodes instead of typing each part onto a job?
  - expected: no guide (the guides do not cover this)
  - returned: `request-and-issue-parts`, `add-products-and-opening-stock`, `frequently-asked-questions`

#### Vector

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`
  - returned: `create-and-manage-work-orders`, `advances-and-bulk-payments`, `booking-job-and-stock-problems`
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`
  - returned: `replenish-parts-for-a-job`, `use-the-bay-board`, `walk-in-to-payment`
- `en-017` does the overall discount on a bill come off the delivery charge too
  - expected: `create-and-issue-an-invoice`
  - returned: `advances-and-bulk-payments`, `record-invoice-payments`, `frequently-asked-questions`
- `en-019` A customer gave me one cheque to cover four old bills. Can I enter it in one go, and which bill gets paid off first?
  - expected: `advances-and-bulk-payments`
  - returned: `create-and-manage-customers`, `walk-in-to-payment`, `start-using-sam`
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`
  - returned: `replenish-parts-for-a-job`, `walk-in-to-payment`, `inspection-to-approved-repair`
- `en-023` A company with lots of vans wants one bill at the end of each month for everything we did, not one per visit. Is that possible?
  - expected: `fleet-billing-and-payments`
  - returned: `frequently-asked-questions`, `start-using-sam`, `walk-in-to-payment`
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`
  - returned: `book-a-service-online`, `walk-in-to-payment`, `sign-in-and-passwords`
- `na-001` Can I link this to QuickBooks so my accountant gets our bills without retyping them?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `frequently-asked-questions`, `walk-in-to-payment`
- `na-003` The same customer got saved twice under two phone numbers. How do I join the two together?
  - expected: no guide (the guides do not cover this)
  - returned: `manage-service-bookings`, `walk-in-to-payment`, `replenish-parts-for-a-job`
- `na-004` I'm opening a second branch next year. How many branches can one account handle?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `start-using-sam`, `use-sami`
- `na-006` Can the customer sign on a tablet when he drops off the car?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `walk-in-to-payment`, `frequently-asked-questions`
- `na-010` Can we scan part barcodes instead of typing each part onto a job?
  - expected: no guide (the guides do not cover this)
  - returned: `replenish-parts-for-a-job`, `create-and-send-an-estimate`, `request-and-issue-parts`

#### Keyword

- `en-001` Ours is installed on our own server. A friend's workshop sees a subscription page in his settings and we don't have one. Is ours broken?
  - expected: `overview`, `plans-trials-and-billing`
  - returned: `start-using-sam`, `company-branding-and-preferences`, `customer-portal`
- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`
  - returned: `create-and-approve-job-orders`, `rework-and-completion`, `overview`
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`
  - returned: `create-and-approve-job-orders`, `rework-and-completion`, `add-employees-and-share-access`
- `en-019` A customer gave me one cheque to cover four old bills. Can I enter it in one go, and which bill gets paid off first?
  - expected: `advances-and-bulk-payments`
  - returned: `first-workshop-job`, `create-and-manage-customers`, `customer-portal`
- `en-021` I took the week's cash out of the drawer and banked it. How do I record that so it doesn't look like extra sales?
  - expected: `transfers-and-supplier-payments`
  - returned: `record-invoice-payments`, `advances-and-bulk-payments`, `review-vat-and-tax-figures`
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`
  - returned: `create-and-approve-job-orders`, `rework-and-completion`, `booking-job-and-stock-problems`
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`
  - returned: `create-and-manage-work-orders`, `payment-accounts-and-ledger`, `review-vat-and-tax-figures`
- `en-032` A customer paid and left last week, and now he's back saying the same noise is still there. Do I open his old job again?
  - expected: `walk-in-to-payment`, `rework-and-completion`
  - returned: `first-workshop-job`, `create-and-manage-customers`, `overview`
- `en-034` The part we were waiting for came in and I booked the delivery, but the store still can't give the mechanic his part. It keeps saying there isn't enough.
  - expected: `replenish-parts-for-a-job`, `request-and-issue-parts`
  - returned: `booking-job-and-stock-problems`, `book-a-service-online`, `manage-service-bookings`
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`
  - returned: `create-and-approve-job-orders`, `overview`, `rework-and-completion`
- `en-040` Can one mechanic clock two jobs together?
  - expected: `frequently-asked-questions`, `track-technician-work`, `booking-job-and-stock-problems`
  - returned: `rework-and-completion`, `overview`, `create-and-approve-job-orders`
- `na-001` Can I link this to QuickBooks so my accountant gets our bills without retyping them?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `email-whatsapp-and-link-problems`, `fleet-billing-and-payments`
- `na-003` The same customer got saved twice under two phone numbers. How do I join the two together?
  - expected: no guide (the guides do not cover this)
  - returned: `create-and-manage-customers`, `first-workshop-job`, `create-and-manage-work-orders`
- `na-004` I'm opening a second branch next year. How many branches can one account handle?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `overview`, `booking-job-and-stock-problems`
- `na-006` Can the customer sign on a tablet when he drops off the car?
  - expected: no guide (the guides do not cover this)
  - returned: `create-and-manage-customers`, `first-workshop-job`, `create-and-send-an-estimate`
- `na-010` Can we scan part barcodes instead of typing each part onto a job?
  - expected: no guide (the guides do not cover this)
  - returned: `create-and-approve-job-orders`, `understand-job-costs`, `replenish-parts-for-a-job`

#### Hybrid

- `en-003` I got an alert about a job waiting for me, but when I opened it somebody had already dealt with it. Is the alert wrong?
  - expected: `dashboard-and-navigation`
  - returned: `booking-job-and-stock-problems`, `rework-and-completion`, `create-and-approve-job-orders`
- `en-010` Hours lost if I take a mechanic off a job halfway?
  - expected: `track-technician-work`
  - returned: `rework-and-completion`, `create-and-approve-job-orders`, `maintenance-and-driver-handovers`
- `en-011` My mechanic left with two small items still open, but I looked at the car myself and it's done. How do I close the job without him?
  - expected: `rework-and-completion`
  - returned: `first-workshop-job`, `replenish-parts-for-a-job`, `dashboard-and-navigation`
- `en-022` We send gearbox rebuilds out to another shop. How do I put what they charge us onto the job for that car?
  - expected: `understand-job-costs`
  - returned: `replenish-parts-for-a-job`, `create-and-approve-job-orders`, `create-and-send-an-estimate`
- `en-026` I sent a new guy his invite on Monday. He only opened it on Thursday and now it says it's no good. Why?
  - expected: `add-employees-and-share-access`, `sign-in-and-passwords`, `login-access-and-plan-problems`, `email-whatsapp-and-link-problems`
  - returned: `create-and-manage-work-orders`, `set-booking-hours-and-holidays`, `booking-job-and-stock-problems`
- `en-037` I clicked to turn an appointment into a job and the page just hung. Did it go through, or should I click again?
  - expected: `booking-job-and-stock-problems`, `manage-service-bookings`
  - returned: `rework-and-completion`, `book-a-service-online`, `create-and-approve-job-orders`
- `na-001` Can I link this to QuickBooks so my accountant gets our bills without retyping them?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `print-and-share-documents`, `email-whatsapp-and-link-problems`
- `na-003` The same customer got saved twice under two phone numbers. How do I join the two together?
  - expected: no guide (the guides do not cover this)
  - returned: `share-progress-and-inspections`, `create-and-manage-work-orders`, `manage-service-bookings`
- `na-004` I'm opening a second branch next year. How many branches can one account handle?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `booking-job-and-stock-problems`, `fleet-accounts-and-drivers`
- `na-006` Can the customer sign on a tablet when he drops off the car?
  - expected: no guide (the guides do not cover this)
  - returned: `walk-in-to-payment`, `share-progress-and-inspections`, `manage-service-bookings`
- `na-010` Can we scan part barcodes instead of typing each part onto a job?
  - expected: no guide (the guides do not cover this)
  - returned: `replenish-parts-for-a-job`, `create-and-approve-job-orders`, `request-and-issue-parts`

### Misses on dev

#### Title-picking

- `na-002` Is there an app for the mechanics' phones?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `frequently-asked-questions`, `track-technician-work`
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide (the guides do not cover this)
  - returned: `plans-trials-and-billing`, `overview`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide (the guides do not cover this)
  - returned: `payment-accounts-and-ledger`, `record-invoice-payments`, `frequently-asked-questions`
- `na-008` Will it text my customers on its own when their next oil change is coming up?
  - expected: no guide (the guides do not cover this)
  - returned: `messaging-and-campaigns`, `maintenance-and-driver-handovers`, `use-sami`
- `na-009` Does it make the WPS file for paying staff wages through the bank?
  - expected: no guide (the guides do not cover this)
  - returned: `frequently-asked-questions`, `overview`

#### Vector

- `en-002` We're brand new on this. What has to be sorted out in the system before the first real car comes in?
  - expected: `first-workshop-job`
  - returned: `replenish-parts-for-a-job`, `walk-in-to-payment`, `booking-job-and-stock-problems`
- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`
  - returned: `frequently-asked-questions`, `start-using-sam`, `financial-reports-and-statements`
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`
  - returned: `overview`, `walk-in-to-payment`, `share-progress-and-inspections`
- `en-024` The form our drivers fill in when they swap shifts won't take the kilometre figure. What's wrong?
  - expected: `maintenance-and-driver-handovers`
  - returned: `booking-job-and-stock-problems`, `roles-and-permissions`, `replenish-parts-for-a-job`
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`
  - returned: `payment-accounts-and-ledger`, `fleet-billing-and-payments`, `create-service-packages`
- `en-031` A customer can't get into the page where he looks at his own bills. He already tried the forgotten-password option we use as staff, and that didn't help.
  - expected: `customer-portal`
  - returned: `roles-and-permissions`, `company-branding-and-preferences`, `login-access-and-plan-problems`
- `en-036` Since this morning we can open and read everything, but nothing we change will save. Nobody in the shop can.
  - expected: `login-access-and-plan-problems`, `plans-trials-and-billing`
  - returned: `booking-job-and-stock-problems`, `roles-and-permissions`, `replenish-parts-for-a-job`
- `en-038` I want to bill a repair we just finished, but the job isn't in the list when I go to invoice it.
  - expected: `invoice-and-payment-problems`, `create-and-issue-an-invoice`
  - returned: `walk-in-to-payment`, `replenish-parts-for-a-job`, `inspection-to-approved-repair`
- `na-002` Is there an app for the mechanics' phones?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `add-and-manage-vehicles`, `maintenance-and-driver-handovers`
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide (the guides do not cover this)
  - returned: `walk-in-to-payment`, `replenish-parts-for-a-job`, `overview`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide (the guides do not cover this)
  - returned: `frequently-asked-questions`, `walk-in-to-payment`, `plans-trials-and-billing`
- `na-008` Will it text my customers on its own when their next oil change is coming up?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `add-employees-and-share-access`, `share-progress-and-inspections`
- `na-009` Does it make the WPS file for paying staff wages through the bank?
  - expected: no guide (the guides do not cover this)
  - returned: `advances-and-bulk-payments`, `add-employees-and-share-access`, `fleet-accounts-and-drivers`

#### Keyword

- `en-002` We're brand new on this. What has to be sorted out in the system before the first real car comes in?
  - expected: `first-workshop-job`
  - returned: `overview`, `company-branding-and-preferences`, `start-using-sam`
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`
  - returned: `create-and-manage-customers`, `manage-service-bookings`, `first-workshop-job`
- `en-008` Customer said no to my quote yesterday, then rang today and wants the work done after all. Do I have to write a new quote?
  - expected: `create-and-send-an-estimate`, `frequently-asked-questions`
  - returned: `overview`, `create-and-manage-work-orders`, `create-and-manage-customers`
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`
  - returned: `overview`, `sign-in-and-passwords`, `dashboard-and-navigation`
- `en-020` The electricity bill came today but I'll only pay it next month. How do I log it now so it isn't forgotten?
  - expected: `expenses-and-other-income`
  - returned: `overview`, `booking-job-and-stock-problems`, `fleet-billing-and-payments`
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`
  - returned: `create-service-packages`, `payment-accounts-and-ledger`, `record-invoice-payments`
- `en-035` My receptionist can see the online appointments but I want her to be able to accept them too. Where do I change what she's allowed to do?
  - expected: `roles-and-permissions`
  - returned: `record-invoice-payments`, `book-a-service-online`, `set-booking-hours-and-holidays`
- `en-036` Since this morning we can open and read everything, but nothing we change will save. Nobody in the shop can.
  - expected: `login-access-and-plan-problems`, `plans-trials-and-billing`
  - returned: `create-job-templates`, `first-workshop-job`, `roles-and-permissions`
- `na-002` Is there an app for the mechanics' phones?
  - expected: no guide (the guides do not cover this)
  - returned: `create-and-manage-customers`, `email-whatsapp-and-link-problems`, `book-a-service-online`
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `customer-portal`, `start-using-sam`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide (the guides do not cover this)
  - returned: `record-invoice-payments`, `fleet-billing-and-payments`, `advances-and-bulk-payments`
- `na-008` Will it text my customers on its own when their next oil change is coming up?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `create-and-manage-customers`, `first-workshop-job`
- `na-009` Does it make the WPS file for paying staff wages through the bank?
  - expected: no guide (the guides do not cover this)
  - returned: `login-access-and-plan-problems`, `overview`, `sign-in-and-passwords`

#### Hybrid

- `en-004` One of my regulars always pays about a month after he gets the bill. Can the due date fill itself in for him?
  - expected: `create-and-manage-customers`
  - returned: `financial-reports-and-statements`, `maintenance-and-driver-handovers`, `overview`
- `en-005` Does a returning customer have to type in all his car details again when he books through our link?
  - expected: `book-a-service-online`
  - returned: `manage-service-bookings`, `share-progress-and-inspections`, `walk-in-to-payment`
- `en-012` Which car is on which lift - can that screen go up on a big TV?
  - expected: `use-the-bay-board`
  - returned: `overview`, `inspect-a-vehicle`, `sign-in-and-passwords`
- `en-028` I paid for the upgrade and the money has left my card, but our account still shows the old package.
  - expected: `plans-trials-and-billing`
  - returned: `payment-accounts-and-ledger`, `create-service-packages`, `record-invoice-payments`
- `na-002` Is there an app for the mechanics' phones?
  - expected: no guide (the guides do not cover this)
  - returned: `messaging-and-campaigns`, `manage-service-bookings`, `book-a-service-online`
- `na-005` What do you charge to come out to my garage and get everything set up for us?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `start-using-sam`, `create-service-packages`
- `na-007` Can our card machine be hooked up so a card payment shows on the bill by itself?
  - expected: no guide (the guides do not cover this)
  - returned: `record-invoice-payments`, `frequently-asked-questions`, `fleet-billing-and-payments`
- `na-008` Will it text my customers on its own when their next oil change is coming up?
  - expected: no guide (the guides do not cover this)
  - returned: `share-progress-and-inspections`, `walk-in-to-payment`, `create-and-manage-customers`
- `na-009` Does it make the WPS file for paying staff wages through the bank?
  - expected: no guide (the guides do not cover this)
  - returned: `overview`, `roles-and-permissions`, `plans-trials-and-billing`

## Label corrections

None. No correction is logged in evaluation/corrections.jsonl.

## Limits

- The set is small. In `test`, one English question moves hit@1 or hit@3 by about 5 points (1 of about 20). One Arabic question, or one unanswerable question, moves its result by about 20 points (1 of about 5).
- The threshold is set from about five unanswerable `dev` questions and checked on about five more. The refusal numbers show whether refusing works at all, not how often. 10 of the 50 scored questions are unanswerable because the set was built that way; the threshold depends on that mix.
- One threshold serves both languages and is fitted mostly on English questions. Arabic wrong refusals show what that costs. Refusing Arabic questions that the guides do not cover is not measured.
- Keyword mode has no threshold, so its refusal numbers follow from question wording and language, not from tuning. Vector and hybrid share one refusal rule, so their refusal numbers are the same by construction.
- The questions are drafted by a model that has read the guides. They may echo guide wording more than real customers would, which flatters keyword and vector search. The drafting rules and the review reduce this; they do not remove it.
- The expected guides are one reviewer's labels, and the reviewer also built the search. A missing acceptable guide turns a right result into a miss.
- Title-picking here is one choice from id, slug and title. An assistant that reads a guide before choosing the next may do better, so beating this baseline is not the same as beating such an assistant. The slug is shown as well as the title, and many slugs contain words the title does not, so the baseline has slightly more than titles to go on. Its model is not the newest available; a newer one might pick better. It was run once, and a second run could move its numbers by a question or two.
- A guide-level hit for search does not mean the returned passages hold the answer; passage hit@3 is there to show the difference. Text handed over is only like for like when read beside it.
- Arabic questions are matched against English passages. Published benchmarks for this embedding model show that this is clearly weaker than English to English, so lower Arabic results are expected. The baseline's model reads Arabic directly and has no such handicap.

The points above about Arabic results do not apply to this report yet: no Arabic question has been reviewed.
