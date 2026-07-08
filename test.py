import sys
sys.path.insert(0, r"d:\E&S Solutions\Survey-analysis")
import pandas as pd
import charts
import stats

OUT = r"C:\Users\ashis\AppData\Local\Temp\claude\d--E-S-Solutions-Survey-analysis\771cc742-671b-4f11-a260-070cf6bceab8\scratchpad"

df = pd.read_csv(f"{OUT}\\llm_extracted_analysis.csv")
members = pd.read_csv(f"{OUT}\\llm_extracted_members.csv")

# 1. Bar chart from a real frequency result
freq = stats.frequency(df, "village")
bar = charts.bar_chart(freq.table, "village", "count", title="Households by village")
png = charts.chart_to_png(bar)
with open(f"{OUT}\\chart_bar.png", "wb") as f:
    f.write(png)
print(f"bar chart PNG: {len(png)} bytes")

# 2. Pie chart from the same frequency result
pie = charts.pie_chart(freq.table, "village", "count", title="Household distribution by village")
png = charts.chart_to_png(pie)
with open(f"{OUT}\\chart_pie.png", "wb") as f:
    f.write(png)
print(f"pie chart PNG: {len(png)} bytes")

# 3. Demographic pyramid from real member data
pyramid = charts.demographic_pyramid(members)
png = charts.chart_to_png(pyramid)
with open(f"{OUT}\\chart_pyramid.png", "wb") as f:
    f.write(png)
print(f"pyramid chart PNG: {len(png)} bytes")

print("\nAll three chart types rendered and exported to PNG successfully.")
