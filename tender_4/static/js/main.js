// static/js/main.js
document.addEventListener("DOMContentLoaded", function() {
    // Load historical data and populate the Monitor table.
    async function loadHistoricalData() {
      try {
        const response = await fetch('/api/historical');
        const data = await response.json();
        const tbody = document.querySelector("#historicalTable tbody");
        tbody.innerHTML = "";
        data.forEach(event => {
          const row = document.createElement("tr");
          row.innerHTML = `
            <td>${event.event_id}</td>
            <td>${event.commodity}</td>
            <td>${event.item}</td>
            <td>${event.supplier_name}</td>
            <td>${event.bid_amount}</td>
            <td>${event.bid_status}</td>
            <td>${event.submission_timestamp}</td>
            <td>${event.evaluation_score}</td>
          `;
          tbody.appendChild(row);
        });
      } catch (error) {
        console.error("Error loading historical data:", error);
      }
    }
    
    // Handle the Compare Quote form submission.
    document.getElementById("compareForm").addEventListener("submit", async function(e) {
      e.preventDefault();
      const currentEvent = {
        event_id: document.getElementById("cmp_event_id").value,
        commodity: document.getElementById("cmp_commodity").value,
        item: document.getElementById("cmp_item").value,
        line_item: document.getElementById("cmp_line_item").value,
        line_number: document.getElementById("cmp_line_number").value,
        supplier_id: document.getElementById("cmp_supplier_id").value,
        supplier_name: document.getElementById("cmp_supplier_name").value,
        bid_amount: document.getElementById("cmp_bid_amount").value,
        bid_status: document.getElementById("cmp_bid_status").value,
        evaluation_score: document.getElementById("cmp_evaluation_score").value,
        submission_timestamp: document.getElementById("cmp_submission_timestamp").value
      };
      
      try {
        const response = await fetch("/api/ai_insights", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ current_event: currentEvent })
        });
        const result = await response.json();
        const insightsDiv = document.getElementById("insightsResult");
        insightsDiv.style.display = "block";
        insightsDiv.innerHTML = "<strong>AI Insights:</strong><br>" + result.ai_insights;
        console.log("GPT Prompt (for debugging):", result.gpt_prompt);
      } catch (error) {
        console.error("Error getting AI insights:", error);
      }
    });
    
    loadHistoricalData();
});
