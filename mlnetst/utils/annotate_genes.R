cat("[R] I'm in the annotate_genes.R file")
# Importing libraries
suppressMessages(library(AnnotationDbi))
suppressMessages(library(dplyr))
suppressMessages(library(EnsDb.Mmusculus.v79))

args = commandArgs(trailingOnly=TRUE)
parse_args <- function(args) {
    result <- list()
    for(arg in args) {
        if (startsWith(arg, "--")) {
            splitArg <- strsplit(substring(arg, 3), "=")[[1]]
            result[[splitArg[1]]] <- splitArg[2]
        }
    }
    return(result)
}
parsed_args <- parse_args(args)
input_file_path <- parsed_args[["input_file_path"]]
output_file_path <- parsed_args[["output_file_path"]]

# Read the input file
genes_str = readLines(input_file_path)
genes = unlist(strsplit(genes_str, ";"))
genes <- trimws(genes) # Remove leading and trailing whitespace

keytypes_list_end <- AnnotationDbi::keytypes(EnsDb.Mmusculus.v79::EnsDb.Mmusculus.v79)
all_mappings <- list()

# Loop through each keytype and get the mappings
cat("[R] Step 1: Mapping genes to database...\n")
for (keytype in keytypes_list_end){
    tryCatch({
        mapped_ids <- AnnotationDbi::mapIds(
            EnsDb.Mmusculus.v79::EnsDb.Mmusculus.v79,
            keys = genes,
            column = "GENENAME",
            keytype = keytype,
            multiVals = "first"
        )
    all_mappings[[keytype]] <- mapped_ids
    }, error = function(e) {

    })
}
ids <- dplyr::bind_cols(all_mappings)
annotated_df <- AnnotationDbi::select(
    EnsDb.Mmusculus.v79::EnsDb.Mmusculus.v79,
    keys = genes,
    columns = c("GENEBIOTYPE","SYMBOL","SEQNAME"),
    keytype = "SYMBOL"
    )

protein_coding_genes <- annotated_df %>%
  dplyr::distinct( SYMBOL, .keep_all = TRUE) #in database there are duplicated values of gene symbol, they differ in SEQNAME

# Save the results of a temporary file on the folder output/runtime of the project
cat("[R] Step 2: Saving gene annotations to file...\n")
write.table(protein_coding_genes, file = output_file_path, sep = ";",
    quote = FALSE, row.names = FALSE)