// It is unclear to me whether it would be best to compute the generated quantities
// in Stan or in Python. 
// NOTE: this file could just be omitted
matrix[n_bins + 1, n_bins + 1] Omega = multiply_lower_tri_self_transpose(L_Omega);
