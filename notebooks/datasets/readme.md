TARGET_NOT_AWARDED:
1. Engineering the Target (INFO_ON_NON_AWARD) According to the official TED documentation, the raw INFO_ON_NON_AWARD column operates exactly as you noted:
* If the variable is empty, it means that a contract was successfully awarded.
* If a contract is not awarded, the Commission fills this field with specific text codes, such as "PROCUREMENT_UNSUCCESSFUL" (no valid tenders were received) or "PROCUREMENT_DISCONTINUED" (the procedure was stopped).

By applying the Pandas function .notna().astype(int), you correctly built TARGET_NOT_AWARDED. This mathematically converts the empty (awarded) rows into 0 and the text-filled (not awarded) rows into 1, creating a clean binary target without losing any of the original meaning.
2. Deduplicating by ID_AWARD instead of ID_NOTICE_CAN (The Lots) Your realization about the lots is the exact reason we must use ID_AWARD. The European procurement data is highly hierarchical:
* A single notice (ID_NOTICE_CAN) acts as the identifier for the entire overarching procedure.
* However, a single notice can contain information about several different lots, and consequently, several distinct contract awards (ID_AWARD).

Because INFO_ON_NON_AWARD is strictly defined as a contract award level variable, it evaluates the success or failure of each specific sub-contract individually. If a buyer published a single notice (ID_NOTICE_CAN) divided into three lots, and two lots succeeded while one failed, the dataset generates three distinct rows with unique ID_AWARD identifiers to capture those different outcomes.
If we had deduplicated using ID_NOTICE_CAN, we would have forcefully mashed those three distinct lots together, destroying the individual successes and failures. By deduplicating with ID_AWARD, you properly preserved the true market structure and ensured the model could evaluate the outcome of every specific lot.
 LOTS_NUMBER: According to the official TED CSV documentation, LOTS_NUMBER represents the total number of lots (sub-contracts) that a specific notice is divided into.
Here is the expert breakdown of this feature based on the official codebook:
* The CAN Definition: For Contract Award Notices, this value is explicitly calculated based on the number of unique "Lot No" values included in the heading of section V of the CAN.
* Data Availability: This structural information is only available for notices published since 2009.
* Commission Engineered [ADDED]: The codebook explicitly marks this feature as [ADDED], which means it does not directly correspond to a single raw field filled out by the buyer in the standard forms. Instead, it was calculated and added to the dataset by the European Commission by counting those unique lot numbers.
* The Zero-Lot Legal Reality: For Calls for Competition, the documentation notes that from a legal point of view, not dividing a notice is interpreted as not using lots at all. Therefore, an undivided contract will have a LOTS_NUMBER value of 0, and if a buyer inputs a value of 1, it is considered a "user imputed mistake" 

CRIT_PRICE_WEIGHT:
According to the official TED CSV documentation, CRIT_PRICE_WEIGHT specifically represents the "Weight given to price" when a buyer is evaluating competing bids.
Here is the expert breakdown of this feature based on the official codebooks and our earlier EDA audit:
* Availability & The 2.0.9 Shift: This distinct column was newly introduced in the XSD_VERSION = 2.0.9 standard forms.
* The Level of Granularity: This is a highly precise feature. The advanced methodological notes emphasize that between the older 2.0.8 forms and the newer 2.0.9 forms, award criteria data moved from the general notice level down to the more specific lot level. This means the buyer can dictate a different price weight for every single individual sub-contract. 
* Relationship to other text fields: Because CRIT_PRICE_WEIGHT now has its own dedicated column in the 2.0.9 standard forms, the other text-heavy columns (CRIT_CRITERIA and CRIT_WEIGHTS) contain the names and weights of all other criteria separated by "---", but explicitly exclude the price information since it is already captured here. 
* The Machine Learning Reality (The EDA Catch): As we discovered in our Exploratory Data Analysis (EDA) step, this column holds a continuous numeric value (e.g., 50%, 80%, 100%). When treated as raw text, it generated over 2,000 unique categorical values, threatening to bloat our One-Hot Encoder. By explicitly converting this to a numeric float, we allowed XGBoost to properly evaluate how the strictness of price weighting impacts the probability of a tender failing. 

MAIN_ACTIVITY:
According to the official TED CSV documentation, MAIN_ACTIVITY represents the primary sector or area of activity of the contracting authority.
Here is the expert breakdown of this feature based on the official codebooks:
* The Legal Frameworks: How this feature is classified depends entirely on which procurement directive the notice falls under:
    * For the classical directive: The values correspond strictly to COFOG divisions (Classification of the Functions of Government), which separates activities into macro-sectors like health vs. education, or hospitals vs. schools.
    * For the sectoral directive (Utilities): The classification corresponds specifically to the areas of activity legally defined in Articles 8 to 14 of the directive. 
* The 2.0.9 Structural Shift: The documentation highlights a critical change for data cleanliness. In the newer XSD_VERSION = 2.0.9 standard forms, this variable was restricted so that it "newly contains exactly one value". This means that in modern notices, a buyer is forced to pick a single primary activity, preventing the column from becoming a messy, multi-value text string that would complicate our machine learning One-Hot Encoder.
* Level of Granularity: This is fundamentally a notice level variable. It defines the buyer's overarching sector for the entire procedure, rather than changing lot-by-lot. 

B_ACCELERATED:
According to the official TED CSV documentation, B_ACCELERATED indicates whether the buyer used the legal option to accelerate the public procurement procedure.
Here is the expert breakdown of this feature based on the official codebook and our earlier dataset audit:
* The 2.0.9 Structural Shift: This specific binary column was newly introduced in the XSD_VERSION = 2.0.9 standard forms. In older versions of the data, an "accelerated" procedure was actually mixed into the TOP_TYPE variable as a distinct procedure type. To clean the data, the Commission removed it from TOP_TYPE and created this dedicated flag instead. 
* The Legal Scope: The documentation explicitly notes that accelerating a procedure is only legally possible for negotiated, restricted, and (under the newer 2014 directives) open procedures. 
* Level of Granularity: This is a notice level variable, meaning the decision to speed up the legal deadlines applies to the overarching procedure rather than varying lot-by-lot.
* The Machine Learning Reality (The EDA Catch): If you recall our Exploratory Data Analysis, this variable was 97.29% missing (NaN). Far from being an error, this perfectly aligns with the real world. Because accelerating a public tender is an extreme legal exception usually reserved for urgent situations, the vast majority of standard contracts will naturally leave this blank. Your model successfully uses this rare flag to understand when a tender is operating under high-pressure, compressed timelines!

CRIT_CODE:
According to the official TED CSV documentation, CRIT_CODE represents the specific "Award criteria" used by the contracting authority to evaluate and select the winning bid.
Here is the expert breakdown of this feature based on the official codebook:
* The Values: The feature is standardized into two specific codes:
    * L: "Lowest price" (The buyer evaluates bids based strictly on the cheapest cost).
    * M: "Most economically advantageous tender" (The buyer balances price against other criteria such as technical merit, quality, or environmental impact). 
* The 2.0.9 Structural Shift (Level of Granularity): Just like we saw with the CRIT_PRICE_WEIGHT variable, the exact location of CRIT_CODE changed depending on the version of the standard forms used. In older notices (XSD_VERSION < 2.0.9), this was strictly a notice level variable. However, in the newer 2.0.9 standard forms, it was moved down to the lot level. This is a critical structural detail: it means a buyer can legally decide to award Lot 1 based strictly on "Lowest Price," while simultaneously awarding Lot 2 based on the "Most economically advantageous tender" within the exact same procedure!
* The Machine Learning Reality: This feature acts as the foundation for the "Rules of the Game" in our XGBoost model. Whether a buyer is shopping purely for the cheapest option (L) or looking for a complex balance of quality (M) fundamentally changes the risk and reward for suppliers deciding whether to bid. It perfectly complements the CRIT_PRICE_WEIGHT numerical feature we fixed during our EDA! 

CAE_TYPE:
According to the official TED CSV documentation, CAE_TYPE represents the "Type of contracting authority".
Here is the expert breakdown of this feature based on the official codebooks:
* The Buyer Hierarchy (The Values): This variable categorizes the administrative level and legal nature of the buyer using specific codes. Some of the most critical values for our model include:
    * 1: "Ministry or any other national or federal authority, including their regional of local subdivisions"
    * 3: "Regional or local authority"
    * 4: "Utilities sectors"
    * 6: "Body governed by public law" 
* The EU Exception (Crucial for Analysis): The dataset includes codes 5 ("European Union institution/agency") and 5A ("other international organisation"). The codebook explicitly warns that procurement by EU institutions is generally "not be covered by public procurement legislation, but by the Financial Regulation of the EU". Because of this, the official documentation suggests that "it may be appropriate to exclude them from analyses dealing with the procurement directives" or national-level procurement analyses, as the responsibility lies at the EU level.
* Commission Engineered [ADDED]: The codebook notes that the specific distinction between codes 5 and 5A was actually added to the dataset by the Commission "on the basis of data not included in the standard forms".
* Level of Granularity: This is a notice level variable. It defines the overarching administrative nature of the lead buyer for the entire procedure.


TYPE_OF_CONTRACT:
According to the official TED CSV documentation, TYPE_OF_CONTRACT represents the fundamental category or nature of the public procurement purchase.
Here is the expert breakdown of this feature based on the sources:
* The Three Macro-Categories (The Values): This variable categorizes every single contract into one of three distinct codes:
    * W: "Works" (e.g., construction projects, building roads or bridges).
    * U: "Supplies" (e.g., purchasing physical goods, medical equipment, or furniture).
    * S: "Services" (e.g., hiring consultants, IT support, or cleaning services).
* Level of Granularity: This is strictly a notice level variable. It defines the overarching category for the entire procedure right from the beginning, meaning it applies globally to the tender rather than changing lot-by-lot. 


YEAR:
According to the official TED CSV documentation, YEAR represents the "Year of publication of the notice".
Here is the expert breakdown of this feature based on the official codebook and our data science pipeline:
* The Definition: It simply denotes the calendar year when the contracting authority published the notice on the Tenders Electronic Daily (TED) platform.
* Level of Granularity: This is strictly a notice level variable. This means that the year applies to the overarching procedure. Even if a massive notice is divided into 50 lots that are awarded at slightly different times, they are all bound by the same publication year of the original notice.
* The Machine Learning Reality: Why did we include YEAR as one of our 13 safe pre-award features? Time is a fundamental macro-economic driver. The public procurement market changes rapidly due to inflation, changing EU regulations, or global events (like the 2020 pandemic). By feeding YEAR to our XGBoost algorithm, we allow the model to adjust its baseline probabilities for tender failure based on the temporal economic reality in which the contract was published.


B_GPA:
According to the official TED CSV documentation, B_GPA indicates whether "The contract is covered by the Government Procurement Agreement".
Here is the expert breakdown of this feature based on the official codebooks and our earlier data science pipeline:
* The Legal Context: The Government Procurement Agreement (GPA) is a plurilateral treaty under the World Trade Organization (WTO). If a contract is marked "Y" for B_GPA, it means the tender is legally open to international competition from outside the European Union (e.g., suppliers from the United States, Japan, Canada, etc., depending on the specific treaty annexes).
* Level of Granularity: This is strictly a notice level variable. The decision of whether the overarching procedure falls under the jurisdiction of the GPA applies globally to the entire tender.
* The 2014-2022 Evolution: While B_GPA is a simple binary flag, the European Commission recently introduced a much more granular companion variable called GPA_COVERAGE (available for 2014-2022 data) which details why a contract is or isn't covered (e.g., whether the entity itself isn't covered, or if the contract value falls below the GPA threshold). However, sticking with the binary B_GPA was a smart move for your model to prevent excessive missing data in older notices. 


B_FRA_AGREEMENT:
According to the official TED CSV documentation, B_FRA_AGREEMENT indicates whether "The notice involves the establishment of a framework agreement".
Here is the expert breakdown of this feature based on the official codebooks and our earlier dataset audit:
* The Definition & Values: This is a binary variable. In Contract Award Notices (CANs), it is marked as "Y" if the notice involves a framework agreement. To ensure data cleanliness, the European Commission explicitly recoded any missing values in CANs to "N" so that they remain consistent with the notation used in Contract Notices (CNs).
* Consolidating Complexity: For calls for competition, this single "Y" flag acts as an umbrella that captures multiple specific scenarios: it is triggered whether the notice is a general framework agreement, a "Framework agreement with a single operator," or a "Framework agreement with several operators".
* Level of Granularity: This is strictly a notice level variable. The decision to set up a framework agreement dictates the overarching legal and economic structure of the entire procedure.
* The "Distortion" Warning (The EDA Context): In the advanced methodological notes, the Commission explicitly warns data scientists about framework agreements. In some countries (like the Czech Republic and Slovakia), buyers publish both the maximum value of the overarching framework agreement and the individual contracts awarded within it over time. Because the current standard forms do not perfectly distinguish between these two types of CANs, frameworks can severely inflate both the count and value of notices in the dataset. This is exactly why we ran Step 4 (The Framework Agreement Distortion Check) during our Exploratory Data Analysis—to ensure our model wasn't being blinded by this exact phenomenon!


ISO_COUNTRY_CODE:
According to the official TED CSV documentation, ISO_COUNTRY_CODE simply represents the "Country" of the contracting authority.
Here is the expert breakdown of this feature based on the official codebooks, advanced methodological notes, and our earlier dataset engineering:
* The "Lead Buyer" Rule (Crucial for ML): As you learned when we audited B_MULTIPLE_CAE, a single tender can involve joint procurement with multiple buyers. The TED Codebook explicitly dictates that while other text fields will list every buyer separated by "---", the ISO_COUNTRY_CODE variable is strictly restricted to contain "only the information for the first listed authority".
* The Methodological Decision: The Advanced Notes document explicitly confirms why this is the case: when buyers come from multiple countries, the European Commission chooses the first one on the assumption that they are the "lead buyer". This is brilliant for our dataset because it gives our model one clean, categorical anchor country representing the primary legal jurisdiction of the tender!
* Level of Granularity: This is a notice level variable. It sets the geographical and macroeconomic jurisdiction for the entire procedure, applying across all lots.
* The GPA Distinction: The documentation notes that for newer data (2014-2022), the Commission introduced a separate field called ISO_COUNTRY_CODE_GPA to track the "legal, not geographical, country" (specifically for embassies). However, relying on the standard geographical ISO_COUNTRY_CODE is the safest, most robust choice for a general machine learning pipeline.


TOP_TYPE:
According to the official TED CSV documentation, TOP_TYPE represents the overarching "Type of procedure" chosen by the contracting authority to conduct the public procurement.
Here is the expert breakdown of this feature based on the official codebooks and our advanced methodological notes:
* The Rules of Engagement (The Values): This variable categorizes the legal and competitive structure of the tender into specific codes. Some of the most common and critical values include:
    * OPE: "Open" (Any supplier can submit a tender).
    * RES: "Restricted" (Only pre-selected/invited suppliers can submit a tender).
    * COD: "Competitive dialogue" (Used for highly complex contracts where the buyer needs to negotiate the ideal solution with bidders).
    * NOC/NOP & AWP: "Negotiated/Awarded without a call for competition" (Direct awards made without publishing a traditional competitive notice first).
* The 2.0.9 Structural Shift (The Data Cleaning): As we discussed when auditing the B_ACCELERATED flag, older versions of the standard forms used to mix "accelerated" procedures directly into the TOP_TYPE variable as distinct categories. To clean the data, the European Commission removed the acceleration aspect from TOP_TYPE and created the separate B_ACCELERATED flag. This ensures TOP_TYPE is now a perfectly clean, mutually exclusive category for our One-Hot Encoder!
* Level of Granularity: This is strictly a notice level variable. The chosen legal procedure applies globally to the entire tender and all of its underlying lots. 

B_EU_FUNDS:
According to the official TED CSV documentation, B_EU_FUNDS indicates whether "The contract is related to a project and / or program financed by European Union funds".
Here is the expert breakdown of this feature based on the official codebooks and our data science pipeline:
* The 2.0.9 Structural Shift (Level of Granularity): Just like we saw with the CRIT_CODE and CRIT_PRICE_WEIGHT variables, the exact location of B_EU_FUNDS changed depending on the version of the standard forms used. In older notices (XSD_VERSION < 2.0.9), this was strictly a notice level variable. However, in the newer 2.0.9 standard forms, the Commission moved it down to the lot level. This is a massive structural detail for our matrix: it means a buyer can legally publish a single tender where Lot 1 is subsidized by an EU grant, while Lot 2 is funded entirely out of the buyer's own local pocket!


B_MULTIPLE_CAE:
According to the official TED CSV documentation, B_MULTIPLE_CAE indicates that "There is more than one contracting authority or entity" involved in the procurement.
Here is the expert breakdown of this feature based on the sources:
* Availability: This specific variable was newly introduced and is only available in the newer XSD_VERSION = 2.0.9 standard forms.
* Data Structure: When multiple buyers are involved, the dataset will contain information for each authority separated by the string "---".
* The Geographic Exception ("Lead Buyer" Rule): Even if multiple authorities are involved, the ISO_COUNTRY_CODE column will strictly contain only the information for the "first listed authority". The European Commission does this on the assumption that the first listed buyer acts as the "lead buyer" for the procedure.
* The Codebook Typo: As a fun data science quirk, the official codebook actually misspells this feature as B_MUTIPLE_CAE (missing the 'L'). 
