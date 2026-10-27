"""Generates a structured, multi-page SEC Form 10-K Financial Report PDF for Atlas Technologies, Inc.

Used as a ground-truth document fixture for FinanceBench evaluation.
Includes:
- Page 1: Business Overview & Operating Segments
- Page 2: Management Discussion & Analysis (MD&A)
- Page 3: Consolidated Statements of Operations (Income Statement Table)
- Page 4: Consolidated Balance Sheets & Statements of Cash Flows (Tables)
- Page 5: Note 9 — Debt and Financing Arrangements (Footnote with interest rates, covenants)
- Page 6: Note 14 — Commitments, Contingencies, and Legal Risks (Footnote with CapEx/purchase commitments)
"""

import os
import fitz

def create_financebench_10k_pdf(output_path: str = "evaluation/financebench_10k_sample.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = fitz.open()

    pages_data = [
        # PAGE 1
        {
            "title": "PART I — Item 1. Business & Strategic Operating Segments",
            "content": """ATLAS TECHNOLOGIES, INC.
UNITED STATES SECURITIES AND EXCHANGE COMMISSION
FORM 10-K ANNUAL REPORT — FISCAL YEAR ENDED DECEMBER 31, 2024

ITEM 1. BUSINESS OVERVIEW
Atlas Technologies, Inc. (NASDAQ: ATLS) is an enterprise document intelligence and AI infrastructure company founded in 2021. The Company pioneers vectorless hierarchical tree search architectures and autonomous document reasoning engines.

PRIMARY BUSINESS OPERATING SEGMENTS:
1. Atlas Cloud Infrastructure (ACI):
Provides scalable multi-cloud GPU inference nodes, secure managed vectorless storage, and distributed document processing pipelines for Fortune 500 enterprises. ACI generated 58% of consolidated net revenue in fiscal 2024.

2. Intelligent Document Solutions (IDS):
Our flagship software-as-a-service (SaaS) platform delivering hierarchical document indexing, multi-page financial statement extraction, table parsing, and regulatory compliance audit capabilities. IDS generated 32% of total revenue.

3. Edge AI & Autonomous Robotics:
Embedded perception models, edge document intelligence, and real-time inference appliances for industrial robotics and defense clients. This emerging segment contributed 10% of total revenue in fiscal 2024.

STRATEGIC VISION & CORE ADVANTAGES:
The Company's proprietary vectorless retrieval engine eliminates token embedding loss and arbitrary chunk boundaries by parsing hierarchical document trees directly via PageIndex technology. This achieves superior answer groundedness, near-zero hallucination rates, and exact page-level verifiable citations."""
        },

        # PAGE 2
        {
            "title": "PART II — Item 7. Management's Discussion and Analysis (MD&A)",
            "content": """ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS

FISCAL YEAR 2024 EXECUTIVE HIGHLIGHTS:
Fiscal year 2024 was a milestone period of profitable growth and cash generation for Atlas Technologies:
- Total consolidated revenue increased 22.06% year-over-year to $12,450 million in 2024, compared to $10,200 million in 2023 and $8,500 million in 2022.
- Gross margin expanded by 200 basis points to 60.00%, driven by infrastructure cost efficiencies and optimized GPU utilization across our vectorless inference stack.
- Operating income surged 47.73% to $2,863 million, representing an operating margin of 23.00% compared to 19.00% in fiscal 2023.
- Net income reached $2,434 million, delivering diluted earnings per share (EPS) of $4.85, up 44.78% from $3.35 per diluted share in 2023.

CAPITAL ALLOCATION AND R&D INVESTMENTS:
Atlas Technologies continues to prioritize aggressive R&D investment while returning capital to shareholders:
- R&D expenses grew to $2,490 million (20.00% of revenue) focused on PageIndex vectorless search, safe AST numerical reasoning, and multitenant knowledge memory architectures.
- Operating cash flow totaled $3,540 million, supporting capital expenditures of $980 million and generating $2,560 million in Free Cash Flow (105.18% net income conversion).
- Cash, cash equivalents, and marketable securities stood at $5,950 million as of December 31, 2024, providing robust balance sheet liquidity with zero net debt."""
        },

        # PAGE 3
        {
            "title": "PART II — Item 8. Consolidated Statements of Operations",
            "content": """ATLAS TECHNOLOGIES, INC.
CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per share data)

                                              Fiscal Year Ended December 31,
                                                2024       2023       2022
Revenue:
  Product and Infrastructure Revenue          $7,221     $5,916     $4,930
  Subscription and Software Revenue            5,229      4,284      3,570
Total Revenue                                $12,450    $10,200     $8,500

Cost of Revenue:
  Cost of Product and Infrastructure          $3,113     $2,678     $2,295
  Cost of Subscription and Software            1,867      1,606      1,360
Total Cost of Revenue                         $4,980     $4,284     $3,655

Gross Profit                                  $7,470     $5,916     $4,845

Operating Expenses:
  Research and Development (R&D)              $2,490     $2,142     $1,785
  Sales and Marketing (S&M)                    1,494      1,326      1,105
  General and Administrative (G&A)               623        510        425
Total Operating Expenses                      $4,607     $3,978     $3,315

Operating Income                              $2,863     $1,938     $1,530
Interest and Other Income / (Expense), net       $52        $45        $32
Income Before Income Taxes                    $2,915     $1,983     $1,562
Provision for Income Taxes                       481        336        265
Net Income                                    $2,434     $1,647     $1,297

Earnings Per Share (EPS):
  Basic Earnings Per Share                     $5.02      $3.47      $2.79
  Diluted Earnings Per Share                   $4.85      $3.35      $2.69

Weighted-Average Shares Outstanding:
  Basic Shares Outstanding                       485        475        465
  Diluted Shares Outstanding                     502        492        482"""
        },

        # PAGE 4
        {
            "title": "PART II — Item 8. Consolidated Balance Sheets & Statements of Cash Flows",
            "content": """ATLAS TECHNOLOGIES, INC.
CONSOLIDATED BALANCE SHEETS & CASH FLOW HIGHLIGHTS
(In millions, as of December 31, 2024 and 2023)

CONSOLIDATED BALANCE SHEETS:                     December 31, 2024   December 31, 2023
ASSETS
Current Assets:
  Cash and Cash Equivalents                                 $3,850              $2,920
  Marketable Securities                                      2,100               1,800
  Accounts Receivable, net                                   1,420               1,150
  Prepaid Expenses and Other Current Assets                    780                 580
Total Current Assets                                        $8,150              $6,450
Property, Plant and Equipment (PP&E), net                    4,320               3,890
Goodwill and Intangible Assets, net                          3,210               2,650
Total Assets                                               $15,680             $12,990

LIABILITIES AND STOCKHOLDERS' EQUITY
Current Liabilities:
  Accounts Payable                                          $1,120                $980
  Accrued Compensation and Expenses                            890                 780
  Deferred Revenue (Current)                                 1,130               1,020
Total Current Liabilities                                   $3,140              $2,780
Long-Term Debt (carrying amount)                             3,800               3,200
Other Non-Current Liabilities                                1,220               1,000
Total Liabilities                                           $8,160              $6,980
Stockholders' Equity:
  Common Stock and Additional Paid-in Capital                3,980               3,560
  Retained Earnings                                          3,540               2,450
Total Stockholders' Equity                                  $7,520              $6,010
Total Liabilities and Stockholders' Equity                 $15,680             $12,990

STATEMENT OF CASH FLOWS SUMMARY (Fiscal 2024):
  Net Cash Provided by Operating Activities:                $3,540              $2,680
  Capital Expenditures (CapEx):                              $(980)              $(820)
  Free Cash Flow (Operating Cash Flow minus CapEx):         $2,560              $1,860"""
        },

        # PAGE 5
        {
            "title": "PART II — Item 8. Notes to Consolidated Financial Statements: Note 9 — Debt",
            "content": """NOTE 9 — DEBT AND FINANCING ARRANGEMENTS

AGGREGATE DEBT OBLIGATIONS:
As of December 31, 2024, the Company's outstanding debt obligations consisted of the following:

Debt Instrument                     Principal Amount   Carrying Value   Coupon Rate   Maturity Date
Senior Notes (2028 Notes)                    $2,000M          $1,988M         3.75%    October 2028
Convertible Senior Notes (2031 Notes)        $1,800M          $1,812M         1.25%       June 2031
Total Long-Term Debt                         $3,800M          $3,800M         2.82%*

*Represents weighted-average effective interest rate on outstanding fixed-rate debt.

CREDIT FACILITY AND REVOLVING LIQUIDITY:
The Company maintains an unsecured senior revolving credit facility with a syndicate of commercial banks providing borrowing capacity of up to $1,000 million. As of December 31, 2024, there were zero borrowings outstanding under the revolving credit facility, and the full $1,000 million capacity was available.

FINANCIAL COVENANTS AND COMPLIANCE:
Under the terms of the Senior Credit Agreement, Atlas Technologies is required to satisfy specific financial maintenance covenants measured at the end of each fiscal quarter:
1. Minimum Interest Coverage Ratio: Consolidated Adjusted EBITDA to consolidated interest expense must not be less than 3.50 to 1.00. As of December 31, 2024, the Company's interest coverage ratio was 27.20 to 1.00, well in excess of the requirement.
2. Maximum Leverage Ratio: Consolidated total funded debt to Consolidated Adjusted EBITDA must not exceed 3.00 to 1.00. As of December 31, 2024, the leverage ratio was 0.98 to 1.00, maintaining full compliance with all debt covenants.

FAIR VALUE OF FINANCIAL INSTRUMENTS:
The estimated fair market value of the 2028 Senior Notes was $1,970 million and the 2031 Convertible Notes was $1,920 million as of December 31, 2024, determined using Level 2 observable market trading prices."""
        },

        # PAGE 6
        {
            "title": "PART II — Item 8. Notes to Consolidated Financial Statements: Note 14 — Commitments",
            "content": """NOTE 14 — COMMITMENTS, CONTINGENCIES, AND LEGAL PROCEEDINGS

UNCONDITIONAL PURCHASE OBLIGATIONS:
The Company has entered into multi-year unconditional purchase commitments with premier cloud compute and datacenter providers to secure GPU inference clusters and high-bandwidth fiber connectivity. As of December 31, 2024, future minimum purchase commitments were as follows:
- Fiscal 2025: $550 million
- Fiscal 2026: $500 million
- Fiscal 2027: $400 million
Total Unconditional Purchase Commitments through 2027: $1,450 million.

LEGAL PROCEEDINGS AND INTELLECTUAL PROPERTY:
In November 2024, a competitor in the vector database space filed a patent infringement lawsuit against Atlas Technologies in the U.S. District Court for the Northern District of California, alleging infringement of two patents relating to high-dimensional nearest neighbor indexing.
The Company has evaluated the claims in consultation with external counsel and determined that its vectorless hierarchical tree architecture does not utilize nearest-neighbor embedding indices and therefore does not infringe the asserted claims.
The Company estimates the range of possible exposure to be between $25 million and $45 million. However, because management believes the probability of an unfavorable outcome is remote to possible, but not probable, no accrual has been recorded in the consolidated financial statements as of December 31, 2024.

SUBSEQUENT EVENTS:
On January 15, 2025, the Board of Directors authorized a new share repurchase program of up to $1,500 million of the Company's common stock, replacing the prior expired authorization. Repurchases may be executed from time to time in open market or privately negotiated transactions."""
        }
    ]

    for p_idx, p_data in enumerate(pages_data, 1):
        page = doc.new_page(width=612, height=792)  # US Letter
        
        # Header banner
        page.draw_rect(fitz.Rect(36, 24, 576, 44), color=(0.1, 0.2, 0.4), fill=(0.95, 0.96, 0.98))
        page.insert_text(fitz.Point(42, 38), "ATLAS TECHNOLOGIES, INC. | SEC FORM 10-K ANNUAL REPORT", fontsize=9, fontname="helv", color=(0.1, 0.2, 0.4))
        
        # Section title
        page.insert_text(fitz.Point(36, 70), p_data["title"], fontsize=12, fontname="helv", color=(0.05, 0.15, 0.35))
        page.draw_line(fitz.Point(36, 78), fitz.Point(576, 78), color=(0.2, 0.4, 0.7), width=1)
        
        # Body text / Tables
        text_rect = fitz.Rect(36, 92, 576, 740)
        page.insert_textbox(text_rect, p_data["content"], fontsize=8.5, fontname="helv", color=(0.1, 0.1, 0.1))
        
        # Footer
        page.draw_line(fitz.Point(36, 755), fitz.Point(576, 755), color=(0.8, 0.8, 0.8), width=0.5)
        page.insert_text(fitz.Point(42, 768), "Confidential - For Regulatory & Benchmark Evaluation Only", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))
        page.insert_text(fitz.Point(510, 768), f"Page {p_idx} of {len(pages_data)}", fontsize=8, fontname="helv", color=(0.3, 0.3, 0.3))

    # Set document Table of Contents (TOC) for hierarchical tree parsing
    toc = [
        [1, "Item 1. Business & Strategic Operating Segments", 1],
        [1, "Item 7. Management's Discussion and Analysis (MD&A)", 2],
        [1, "Item 8. Consolidated Statements of Operations", 3],
        [1, "Item 8. Consolidated Balance Sheets & Statements of Cash Flows", 4],
        [1, "Note 9. Debt and Financing Arrangements", 5],
        [1, "Note 14. Commitments, Contingencies, and Legal Risks", 6],
    ]
    doc.set_toc(toc)

    doc.save(output_path)
    doc.close()
    print(f"Generated FinanceBench 10-K report PDF at: {output_path} ({len(pages_data)} pages, {len(toc)} TOC sections)")

if __name__ == "__main__":
    create_financebench_10k_pdf()
