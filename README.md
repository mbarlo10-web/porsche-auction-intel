🚗 Porsche 911 Auction Intelligence

ML-Powered Price Advisor for Bring-a-Trailer–Style Auctions
Mark Barlow • MS AIB Candidate, W. P. Carey School of Business

Live App: https://porsche-auction-intel-2vid7hygbu8bbctbfd6.streamlit.app

📌 Overview

This project delivers a fully deployed machine-learning application that predicts Porsche 911 auction sale prices using historical Bring-a-Trailer (BaT) data.

In addition to the public dataset, the Avant-Garde Collection (a-gc.com) provided internal Porsche 911 retail and consignment sales data for model years 2023–2025, enabling real-world validation and cross-market comparison between dealer pricing and auction-based valuations.

This made it possible to test whether the model generalizes beyond auction-only environments and evaluate alignment with premium retail markets.

📊 Key Insights (updated excerpt)

Mileage and age remain the dominant predictors of price.

Submodel rarity (GT3 RS, Turbo S, GT2 RS) shows strong price lift across both auction and retail data.

Seasonality: February–April is consistently stronger across BaT and Avant-Garde’s internal 2023–2025 sales patterns.

Auction behavior vs. dealer sales: Avant-Garde’s dataset showed slightly higher median prices for low-mileage GT and Turbo models relative to auction averages, confirming the model’s ability to generalize to higher retail-value markets.

The model performed well on the 2023–2025 Avant-Garde dataset, validating its external predictive robustness.

📚 Data Sources
Primary Dataset — Auction Data

Harold, F. (2023). Porsche 911 – Prices, Listings & Details (1963–2023).
Kaggle. https://www.kaggle.com/datasets/sidharth178/porsche-911-listings-data

Supplementary Dataset — Dealer Sales Data

Avant-Garde Collection Porsche 911 Sales (2023–2025)
Privately sourced dataset provided for academic modeling, business validation, and test-set evaluation.
Website: https://a-gc.com

🙌 Acknowledgements

Special thanks to:

Avant-Garde Collection for providing exclusive 2023–2025 Porsche 911 sales data and business context that strengthened the model's real-world relevance.
https://a-gc.com

Arizona State University

W. P. Carey School of Business — CIS 508

MLflow, Streamlit, Databricks, and the open-source ML community.
