from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarize_text(text, max_length=150):
    # Prepare the input text in the required format
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(input_ids, max_length=max_length, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


# Example text
text = """
The major cause of rain production is moisture moving along three-dimensional zones of temperature and moisture contrasts known as weather fronts. If enough moisture and upward motion is present, precipitation falls from convective clouds (those with strong upward vertical motion) such as cumulonimbus (thunder clouds) which can organize into narrow rainbands. In mountainous areas, heavy precipitation is possible where upslope flow is maximized within windward sides of the terrain at elevation which forces moist air to condense and fall out as rainfall along the sides of mountains. On the leeward side of mountains, desert climates can exist due to the dry air caused by downslope flow which causes heating and drying of the air mass. The movement of the monsoon trough, or intertropical convergence zone, brings rainy seasons to savannah climes.

The urban heat island effect leads to increased rainfall, both in amounts and intensity, downwind of cities. Global warming is also causing changes in the precipitation pattern, including wetter conditions across eastern North America and drier conditions in the tropics. Antarctica is the driest continent. The globally averaged annual precipitation over land is 715 mm (28.1 in), but over the whole Earth, it is much higher at 990 mm (39 in).[1] Climate classification systems such as the Köppen classification system use average annual rainfall to help differentiate between differing climate regimes. Rainfall is measured using rain gauges. Rainfall amounts can be estimated by weather radar. 
"""
summary = summarize_text(text)
print("Summary:", summary)
