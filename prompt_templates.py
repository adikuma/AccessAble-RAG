# extraction prompt for base ramp compliance parameters
EXTRACTION_PROMPT = """
analyze the combined text below, which contains responses to several specific questions about accessible ramp compliance parameters. your goal is to extract the values for all the requested parameters and populate a json object conforming to the rampcompliance schema.

parameters to extract:
- ramp_run: maximum run length (float, typically meters or feet, convert if needed in mm).
- ramp_landing_length: minimum landing platform size/dimension (float, typically mm or inches, convert if needed).
- ramp_width: minimum clear ramp width (float, typically mm or inches).
- path_width: minimum clear path width (float, typically mm or inches).

combined text (responses to specific queries are separated by '---'):
```
{combined_text}
```

instructions:
- carefully read each response section identified by '--- response for query (parameter_name): ... ---'.
- extract the specific value requested by the query associated with that section.
- associate the extracted value only with the correct parameter field in the final json.
- handle units: prefer mm for lengths/widths if possible, otherwise use the unit given. convert common units (e.g., feet/inches to mm if standard requires). if units are ambiguous, state the value found.
- handle ratios: for slopes like '1:12' or '1 in 12', extract the number (12.0).
- handle ranges: if a range is given (e.g., '1000mm to 1200mm'), use the minimum value unless the query specifies maximum.
- handle missing values: if a specific parameter's value is explicitly stated as not found, not specified, or if the query for it resulted in an error/no response, leave the corresponding field null in the json.
- be precise. do not guess values if they are not present in the text.
- populate all fields in the ramp compliance schema based *only* on the provided combined text.

return *only* the fully populated json object.
"""

# gradient extraction for ramp compliance
GRADIENT_EXTRACTION_PROMPT = """
analyze the text below and extract information about ramp gradients and their corresponding maximum horizontal run lengths.

text to analyze:
{text}

instructions:
- identify ramp gradients mentioned in the text (e.g., 1:12, 1:14, 1:15, etc.)
- for each gradient, extract the EXACT maximum allowed horizontal run length stated in the text
- convert all measurements to millimeters (mm). 1 meter = 1000 mm
- format the output as a JSON object with ONLY this structure:
{{
    "gradient_max_lengths": {{
    "1_12": <max_length>,
    "1_14": <max_length>,
    "1_15": <max_length>,
    ...
    }}
}}
- use ONLY the standardized gradient format: "1_12" for 1:12 gradient
- include ONLY gradients and values EXPLICITLY mentioned in the text
- DO NOT include any other fields or parameters

example:
For the text "For a gradient of 1:12, the maximum length is 6 meters. For 1:14, it is 9 meters."
The correct output is:
{{
"gradient_max_lengths": {{
    "1_12": 6000.0,
    "1_14": 9000.0
}}
}}

return ONLY the JSON object with no explanations.
"""

# clause identification prompt
CLAUSE_PROMPT = """
you are a specialized clause identification agent. your task is to find the exact clause references in accessibility standards that match the given information.

given:
1. a query: the original user's question about accessibility
2. a response: the answer provided from the knowledge base

extract:
1. the document name (e.g., "bca 2019", "as 1428.1-2009") 
2. the specific clause number (e.g., "d3.3(a)(i)", "7.1.1")
3. the section title if available

format your response as a json with:
{
  "document": "document name",
  "clause": "specific clause number",
  "section": "section title (if available)",
  "full_citation": "document name (clause number)"
}

if you cannot find a specific clause reference, return:
{
  "document": null,
  "clause": null,
  "section": null, 
  "full_citation": null
}

always be precise and only return clauses that explicitly match the information in the response.
"""