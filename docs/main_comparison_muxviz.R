#!/usr/bin/env Rscript

# Load required libraries
library(muxViz)
library(readr)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 1) {
  cat("Usage: Rscript main_comparison_muxviz.R <edgelist_file>\n")
  cat("Example: Rscript main_comparison_muxviz.R /path/to/edgelist.csv\n")
  quit(status = 1)
}

edgelist_file <- args[1]

# Check if file exists
if (!file.exists(edgelist_file)) {
  cat("Error: File", edgelist_file, "does not exist.\n")
  quit(status = 1)
}

cat("Reading edge list from:", edgelist_file, "\n")

# Read the pandas DataFrame saved as CSV
tryCatch({
    # Read the edge list (assuming it's saved as CSV from pandas)
    edge_df <- read_csv(edgelist_file, show_col_types = FALSE)
    cat("Successfully loaded edge list with", nrow(edge_df), "edges\n")
    cat("Columns:", paste(colnames(edge_df), collapse = ", "), "\n")

    # Display first few rows
    cat("\nFirst 5 rows of edge list:\n")
    print(head(edge_df, 5))

    # Display what is the indexing of a tibble against a data.frame
    cat("\nEdge df [,1]:\n")
    print(edge_df[, 1])

    # Display summary statistics
    cat("\nSummary statistics:\n")
    print(summary(edge_df))

    n_layers <- max(max(edge_df$layer.from, na.rm = TRUE), max(edge_df$layer.to, na.rm = TRUE))
    n_nodes <-  max(max(edge_df$node.from, na.rm = TRUE), max(edge_df$node.to, na.rm = TRUE))

    cat("\nNumber of layers:", n_layers, "and number of nodes:", n_nodes, "\n")
    # Example muxviz analysis (adjust based on your edge list format)

    isDirected <- TRUE
    cat("\nCreating supra-adjacency matrix...\n")
    edges <- data.frame(
      from = unlist(edge_df[, 1] + n_nodes * (edge_df[, 2] - 1)),
      to = unlist(edge_df[, 3] + n_nodes * (edge_df[, 4] - 1)),
      weight = unlist(edge_df[, 5])
    )

    M <-
      Matrix::sparseMatrix(
        i = edges$from,
        j = edges$to,
        x = edges$weight,
        dims = c(n_nodes * n_layers, n_nodes * n_layers)
      )

    if (sum(abs(M - Matrix::t(M))) > 1e-12 && isDirected == FALSE) {
      message(
        "WARNING: The input data is directed but isDirected=FALSE, I am symmetrizing by average."
      )
      M <- (M + Matrix::t(M)) / 2
    }
    cat("Supra-adjacency matrix created with dimensions:", dim(M), "\n")
    # Compute some centrality measures
    #pr <- GetMultiPageRankCentrality(M, n_layers, n_nodes)
    # Compute degree versatility
    deg <- GetMultiDegree(M, n_layers, n_nodes, isDirected = isDirected)

    cat("\nCentrality measures:\n")
    print(deg)

    # Save processed results (optional)
    output_file <- gsub("\\.csv$", "_analysis.csv", edgelist_file)
    cat("Analysis complete. Results could be saved to:", output_file, "\n")

    }, 
    error = function(e) {
        cat("Error:", e$message, "\n")
        quit(status = 1)
    }
)