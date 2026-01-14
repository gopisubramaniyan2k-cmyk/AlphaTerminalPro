from fpdf import FPDF

def generate_pdf_report(summary_df, filename="portfolio_report.pdf"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Portfolio Risk and Return Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(8)
    pdf.cell(0, 10, "Summary Table:", ln=True)

    pdf.set_font("Arial", size=10)

    # Table Header
    col_width = pdf.w / 5
    pdf.set_fill_color(200, 200, 200)
    for col in summary_df.columns:
        pdf.cell(col_width, 10, col, border=1, ln=0, align="C")
    pdf.ln(10)

    # Table Rows
    for _, row in summary_df.iterrows():
        for item in row:
            pdf.cell(col_width, 10, str(item), border=1, ln=0, align="C")
        pdf.ln(10)

    # Save File
    pdf.output(filename)
    return filename
