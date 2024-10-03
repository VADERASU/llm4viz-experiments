# LLM4Viz
VLAT experiments on ChatGPT-4 and Gemini.  This is based on our TVGC paper "[Do LLMs have Visualization Literacy?](https://www.overleaf.com/project/66eda51b8f10da3134cd7d22)".  <--NEED TO REPLACE LINK

Note: To run the experiments, you will need a "cred.json" file to the repo with the following template:

```
{
  "api-key":"<insert your API key here for ChatGPT>",
  "google-api-key":"<insert your API key here for Gemini>",
}
```
## How to Generate Visualizations

[//]: # (**TODO**: Finish concatenating the code and elaborate, Arlen Fan)
The jupyter notebook for generating visualizations is found at
`visualization creation/Regenerate VLAT.ipynb`

The jupyter notebook contains the code for generating visualizations using `matplotlib` and `plotly`.
Install `!pip install -U kaleido` which installs the kaleido library for exporting visualizations created with plotly

#### From this point on, several visualization types are generated using the following themes described below. All dependent-varaible values are randomly generated. Use different seeds for new charts.

Bar Chart (Matplotlib): A bar chart representing internet speeds for various countries is created using matplotlib. Random values are generated for speeds, and the chart includes labels, title, and formatting.

Line Chart (Matplotlib):
This code generates a line chart of oil prices over the months of a year. It includes formatting for axis labels, title, and grid lines.

100% Stacked Bar Chart (Matplotlib):
This section generates a 100% stacked bar chart using random values for political affiliation (Democrats, Republicans, and Others) across different education levels. It normalizes the values to display percentages.

Treemap (Plotly):
This treemap visualizes the relative sizes of various companies, categorized into sectors (e.g., Search, Social Media, Retail).

Area Chart (Matplotlib):
An area chart for coffee prices over time. It includes custom formatting for the x-axis with dates, and a second x-axis to show the year.

Stacked Bar Chart (Matplotlib):
A stacked bar chart is created, showing room service costs (sandwich, water, peanuts, soda, vodka) for various cities.

Pie Chart (Matplotlib):
A pie chart representing the global smartphone market share is generated with random values for each company. 

Stacked Area Chart (Matplotlib):
This visualization stacks areas for the popularity of girls’ names (Amelia, Isla, Olivia) in the UK.

Scatter Plot (Matplotlib):
A scatter plot showing the relationship between height and weight of 85 males is generated. The plot includes custom formatting for axis labels, titles, and markers, and is saved as an SVG file.

Histogram (Plotly):
A histogram is created to visualize taxi passenger ratings, using random values based on a normal distribution. 

Bubble Chart (Plotly):
A bubble chart is created, displaying metro systems in different cities. The size of each bubble corresponds to the ridership, and the annotations label each city. 

Choropleth Map (Plotly):
A choropleth map visualizes unemployment rates by state in the US. The color scale is customized, and state abbreviations are annotated on the map.


## Instructions for Running Experiments
We conducted five types of experiments: 

1. Evaluate LLMs' Visualization Literacy ([reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BChoices))
2. Examine LLMs' Performance without Vis. ([reVLAT_NoViz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BChoices))
3. Examine LLMs' Choice-Free Performance ([reVLAT_Viz+NoChoices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BNoChoices))
4. Examine LLMs' Choice-Free Performance without Visualization and Choices ([reVLAT_NoViz+NoChoices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BNoChoices))
5. Examine LLMs' Performance of Decontextualized Visualization
   1. Visualization and choices ([Anon_reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Anon_reVLAT_Viz%2BChoices))
   2. No visualization and choices ([Anon_reVLAT_NoViz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Anon_reVLAT_NoViz%2BChoices))
  
All experiments were run with Python scripts using ChatGPT and Gemini's API.

### Evaluate LLMs' Visualization Literacy
To run this experiment, run the reVLAT_ChatGPT.py and reVLAT_Gemini.py Python scripts in [reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BChoices).  The results will be saved in the "[Results](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BChoices/Results)" directory.

### Examine LLMs' Performance without Vis.
To run this experiment, run the reVLAT_ChatGPT_novis.py and reVLAT_Gemini_novis.py Python scripts in [reVLAT_NoViz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BChoices).  The results will be saved in the "[Results](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BChoices/Results)" directory.

### Examine LLMs' Choice-Free Performance
To run this experiment, run the reVLAT_ChatGPT_nochoices.py and reVLAT_Gemini_nochoices.py Python scripts in [reVLAT_Viz+NoChoices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BNoChoices).  The results will be saved in the "[Results](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_Viz%2BNoChoices/Results)" directory.

### Examine LLMs' Choice-Free Performance without Visualization and Choices
To run this experiment, run the reVLAT_ChatGPT_noviz+nochoices.py and reVLAT_Gemini_noviz+nochoices.py Python scripts in [reVLAT_NoViz+NoChoices](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BNoChoices).  The results will be saved in the "[Results](https://github.com/cseto23/LLM4Viz/tree/main/reVLAT_NoViz%2BNoChoices/Results)" directory.

### Examine LLMs' Performance of Decontextualized Visualization
There are two sets of experiments for decontextualized/anonymized visualizations: visualizations present and absent.  All code is stored [here](https://github.com/cseto23/LLM4Viz/tree/main/Anon).

#### Visualizations Present
To run this experiment, run the ChatGPT_test.py and Gemini_test.py Python scripts in [Anon_reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Anon_reVLAT_Viz%2BChoices).  The results will be saved in the "[ChatGPT/reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Results/ChatGPT/reVLAT_Viz%2BChoices)" directory for ChatGPT and "[Gemini/reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Results/Gemini/reVLAT_Viz%2BChoices)" for Gemini.

#### Visualizations Absent
To run this experiment, run the ChatGPT_test.py and Gemini_test.py Python scripts in [Anon_reVLAT_NoViz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Anon_reVLAT_NoViz%2BChoices).  The results will be saved in the "[ChatGPT/reVLAT_NoViz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Results/ChatGPT/reVLAT_NoViz%2BChoices)" directory for ChatGPT and "[Gemini/reVLAT_Viz+Choices](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Results/Gemini/reVLAT_NoViz%2BChoices)" for Gemini.

## Analysis Code
For experiments 1 and 2, we focused on fitting a logistic model to our data.  In order to explore the best logistic regression model, we performed analysis on a variety of hyperparameters, which was done in [fullLogisticModelScript.py](https://github.com/cseto23/LLM4Viz/blob/main/fullLogisticModelScript.py), and the results were stored [here](https://github.com/cseto23/LLM4Viz/tree/main/Analysis/fullLogisticModel).

Using the results of the logistic regression model(s), we tested whether the visualization presence or absence affected the correctness in [vizDiffProbScript.py](https://github.com/cseto23/LLM4Viz/blob/main/vizDiffProbScript.py).  Similarly, we tested whether ChatGPT or Gemini was better at answering VLAT questions in [llmDiffProbScript.py](https://github.com/cseto23/LLM4Viz/blob/main/llmDiffProbScript.py).  The remaining analyses/plot code for experiments 1 and 2 were done in the [Analysis.ipynb](https://github.com/cseto23/LLM4Viz/blob/main/Analysis.ipynb) notebook.

The analysis on experiments 3 and 4 were done in the "[noChoices](https://github.com/cseto23/LLM4Viz/tree/main/Analysis/noChoices)" directory

[//]: # (&#40;**TODO**: Elaborate paragraph, explain different notebooks, Arlen Fan&#41;.)
This directory consists of jupyter notebooks that process the results for the no choices, range-based responses. Starting with `arlen_range_analysis.ipynb`, the data is read df = pd.read_csv('noChoices_range_only.csv')
and then the range is cleaned using string processing. Four separate metrics for determining range overlap/performance are calculated, including percentage overlap, Jaccard Index, Dice-Sørensen, and Overlap Coefficient. The results are synthesized into a DataFrame and outputted in the csv file `range_only_analysis_corrected.csv`

Then natural continuation of this analysis is to analyze the results, using both `range_plots.ipynb` and `range_plots_transponse.ipynb`
We split the dataset by LLM and experiment type:
```angular2html
df_gpt_viz_no_choices = df[(df['LLM'] == 'ChatGPT') & (df['expType'] == 'reVLAT_Viz+NoChoices')]
df_gpt_no_viz_no_choices = df[(df['LLM'] == 'ChatGPT') & (df['expType'] == 'reVLAT_NoViz+NoChoices')]
df_gemini_viz_no_choices = df[(df['LLM'] == 'Gemini') & (df['expType'] == 'reVLAT_Viz+NoChoices')]
```

and create a a new category for Gemini, no viz, no choices:
```angular2html
df_gemini_no_viz_no_choices = pd.DataFrame(columns=df.columns)  # Empty DataFrame
df_gemini_no_viz_no_choices = df_gemini_viz_no_choices.copy()
df_gemini_no_viz_no_choices['expType'] = 'reVLAT_NoViz+NoChoices'
```

We clean the dataset and plot the metrics (percentage overlap, Jaccard Index, Dice-Sørensen, and Overlap Coefficient) as already done in `arlen_range_analysis.ipynb`



The `noChoices_test.ipynb` processes experimental data involving Q&A and calculautes different metrics, handles both text and numerical responses. The results are read from `../../reVLAT_QAs.json`.
Task types are identified and then subsequently sorted
```angular2html
taskTypes = {
    'comparisons': 'make comparisons',
    'correlations': 'find correlations/trends',
    'extremum': 'find extremum',
    'value': 'retrieve value',
    'range': 'determine range'
}
```
Response values are counted and cleaned, for example stripping unnecessary characters and removing "approximately" which is a word that GPT likes to insert for these noChoices questions.
Responses are handled by task type. Fianlly, the respones are checked and a score is given. If both the response and answer are textual, then the code compares them and assigns a correctness value, which is binary. For numerical responses,  it calculates both the absolute error and relative error between the response and the answer.
Then, a confidence interval calcluation is applied. This function calculates 95% confidence intervals for a specified property (e.g., relativeError) in the DataFrame by grouping the data by model (LLM) and experiment type (expType).
After this step, we use matplotlib to create a horizontal bar plot showing the mean and confidence intervals for the specified property.
Another step is done to handle text questions. It filters the results based on these questions, organizing the data for further analysis or visualization.
All of this is outputted in `noChoicesResults_text_singleWord.csv`





Finally, the decontextualized/anonymized experiments were analyzed in the [anonymized_analysis.ipynb](https://github.com/cseto23/LLM4Viz/blob/main/Anon/anonymized_analysis.ipynb) notebook. The results will be saved in the "[Results](https://github.com/cseto23/LLM4Viz/tree/main/Anon/Results)" directory.
