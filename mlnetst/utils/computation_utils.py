def compute_tensor_memory_usage(n, l):
    """
    Calculate expected memory usage for multilayer network tensor.
    
    Args:
        n: number of nodes
        l: number of layers
    
    Returns:
        Memory requirements in bytes and human-readable format
    """
    
    # Main tensor: n × l × n × l elements
    total_elements = n * l * n * l  # This is n² × L²
    bytes_per_element = 4  # float32
    main_tensor_bytes = total_elements * bytes_per_element
    
    # Auxiliary memory (much smaller):
    # - Distance matrix: n × n × 4 bytes
    # - Temporary vectors: ~2n × 4 bytes per layer
    # - Coordinate vectors: 2n × 4 bytes
    auxiliary_bytes = n * n * 4 + 2 * n * 4 + 2 * n * 4
    
    total_bytes = main_tensor_bytes + auxiliary_bytes
    
    # Convert to human-readable format
    def bytes_to_human(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024
        return f"{bytes_val:.2f} PB"
    
    print(f"Memory Analysis for n={n}, l={l}:")
    print(f"  Total elements in tensor: {total_elements:,}")
    print(f"  Main tensor memory: {bytes_to_human(main_tensor_bytes)}")
    print(f"  Auxiliary memory: {bytes_to_human(auxiliary_bytes)}")
    print(f"  Total expected memory: {bytes_to_human(total_bytes)}")
    print(f"  Main tensor dominates: {100 * main_tensor_bytes / total_bytes:.1f}% of total")
    
    return total_bytes

