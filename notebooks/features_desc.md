ID_NOTICE_CAN
Description: "Unique identifier of the contract award notice / voluntary ex-ante transparency notice"
.
Why we use it: It serves as the primary identifier at the overall notice level
.
ID_AWARD
Description: "Unique contract award identifier"
.
Why we use it: A single notice can contain multiple lots and multiple contract awards. By deduplicating the dataset at the ID_AWARD level rather than the notice level, we perfectly preserve the multi-lot structure of the data without inappropriately dropping valid distinct procurements
,
.
INFO_ON_NON_AWARD
Description: "If the variable is empty, then a contract was awarded. PROCUREMENT_UNSUCCESSFUL means that 'A contract is not awarded' because 'No tenders or requests to participate were received or all were rejected'. PROCUREMENT_DISCONTINUED means that 'A contract is not awarded' because of 'Other reasons (discontinuation of procedure)'"
.
Why we use it: This is our absolute, leak-free target variable (y). It explicitly tells us whether a procedure failed or succeeded without relying on leaky post-award fields like the winner's name
.

Safe Pre-Award Features
We strictly selected these columns because they represent "pre-award" knowledge. Using post-award fields (like NUMBER_OFFERS or WIN_NAME) creates target leakage, allowing the model to cheat by looking at the result to predict the result
,
.
TOP_TYPE
Description: "Type of procedure. The values are the following: COD 'competitive dialogue', OPE 'open', RES 'restricted', INP 'innovative partnership', NOC/NOP 'negotiated without a call for competition', AWP 'award without prior publication of a contract notice'"
.
Why we use it: The type of competition heavily influences the likelihood of failure. Crucially, we use this column to identify and remove direct awards (NOC, NOP, AWP) where the buyer hand-picks a winner without competition, as these artificially inflate the model's success rate
,
,
.
TYPE_OF_CONTRACT
Description: "Type of contract. The values are the following: W 'Works', U 'Supplies', S 'Services'"
.
Why we use it: Procurement dynamics differ drastically based on what is being bought (e.g., buying pencils vs. building a highway).
B_ACCELERATED
Description: "The option to accelerate the procedure has been used. This is possible for negotiated, restricted, and (under the 2014 directives) open procedures"
.
Why we use it: Accelerated timelines can limit bidder participation, impacting the probability of an award.
CRIT_CODE
Description: "Award criteria. The values are the following: L 'Lowest price', M 'Most economically advantageous tender'"
.
Why we use it: Knowing whether a buyer is forced to pick the cheapest option or can weigh quality and technical merit fundamentally changes the competitive landscape
.
CRIT_PRICE_WEIGHT
Description: "Weight given to price"
.
Why we use it: This indicates exactly how much the buyer prioritizes the lowest price. A higher price weight often correlates with simpler procurements that have different success dynamics
.
CPV
Description: "The main Common Procurement Vocabulary code of the main object of the contract"
.
Why we use it: This classifies the specific market sector (e.g., medical equipment, agricultural supplies). We reduce this to a 2-digit division to capture broad industry trends without causing a high-cardinality explosion in our feature matrix
,
.
ISO_COUNTRY_CODE
Description: "'Country' for the first listed authority"
.
Why we use it: Procurement cultures and reporting standards vary wildly across the EU
. For example, the TED documentation explicitly warns that Romania has a specific national system setup that produces duplicates
, and other countries have inconsistent practices for reporting non-awards
. The model uses this to adjust expectations based on geographic data quirks.
B_EU_FUNDS
Description: "The contract is related to a project and / or programme financed by European Union funds"
.
Why we use it: EU-funded projects often come with stricter auditing, longer timelines, and more rigid compliance rules, altering how the procurement behaves.
B_FRA_AGREEMENT
Description: Indicates "The notice involves the establishment of a framework agreement"
.
Why we use it: Framework agreements (umbrella contracts for future purchases) have fundamentally different award dynamics than one-off specific purchases
,
.
B_DYN_PURCH_SYST
Description: "The notice involves contract(s) based on a dynamic purchasing system"
.
Why we use it: Similar to framework agreements, DPS introduces a different structural process for how vendors are continuously onboarded and selected
.
AWARD_EST_VALUE_EURO
Description: "Estimated CA value, in EUR, without VAT"
.
Why we use it: We explicitly chose this over the standard VALUE_EURO column. According to TED calculations, if the final value is missing, standard value fields are filled with the lowest received bid
. Using the lowest bid leaks post-award information. However, as our final Information Gain metric proved, even this "safe" estimated field acts as a massive data leak in the CAN dataset: if a contract is not awarded, the buyer simply leaves the estimated award value entirely blank, allowing the model to achieve high accuracy just by checking if the value is missing
.