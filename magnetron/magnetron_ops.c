/*
** +=======================================================================+
** | (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>                  |
** +=======================================================================+
*/

#include "magnetron_internal.h"

extern mag_tensor_t* mag_tensor_init_internal(mag_ctx_t* ctx, mag_dtype_t type, int64_t rank, const int64_t* shape, mag_tensor_t* view, size_t view_offs);

static mag_tensor_t* mag_tensor_inplace_view(mag_tensor_t* base) {
    return mag_tensor_init_internal(base->ctx, base->dtype, base->rank, base->shape, base, 0);
}

/*
** ###################################################################################################################
** Operator Validation Helpers
** ###################################################################################################################
*/

/* Check if the input tensors are not null and valid. Return true if valid, else false. */
static bool mag_check_are_inputs_valid(mag_op_t op, mag_tensor_t** inputs, uint32_t numin) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_unlikely(meta->input_count != numin || numin > MAG_MAX_OP_INPUTS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            meta->mnemonic, meta->input_count, numin
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < meta->input_count; ++i) {
        if (mag_unlikely(!inputs[i])) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Input tensor %u is NULL.\n"
                "    Hint: Ensure all input tensors are valid and non-NULL.\n",
                meta->mnemonic, i
            );
            mag_print_separator(stderr);
            fputc('\n', stderr);
            fflush(stderr);
            return false;
        }
    }
    return true;
}

/* Check if the op parameters exist and have valid types. Return true if valid, else false. */
static bool mag_check_are_op_params_valid(mag_op_t op, const mag_op_param_t* params, uint32_t numparams) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);

    bool has_required = false;
    for (int i=0; i < MAG_MAX_OP_PARAMS; ++i) { /* If we have no required op params, skip check */
        if (meta->op_param_layout[i].is_required) {
            has_required = true;
            break;
        }
    }
    if (!has_required) return true; /* No required op params, skip check. */

    if (mag_unlikely(numparams > MAG_MAX_OP_PARAMS)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Too many operation parameters provided.\n"
            "    - Maximum: %u\n"
            "    - Provided: %u\n"
            "    Hint: Ensure the correct number of operation parameters are provided.\n",
            meta->mnemonic, MAG_MAX_OP_PARAMS, numparams
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    if (mag_unlikely(!params)) {
        mag_print_separator(stderr);
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation parameters are NULL.\n"
            "    Hint: Ensure all operation parameters are valid and non-NULL.\n",
            meta->mnemonic
        );
        mag_print_separator(stderr);
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < numparams; ++i) {
        if (mag_unlikely(meta->op_param_layout[i].is_required && params[i].type != meta->op_param_layout[i].type)) {
            mag_print_separator(stderr);
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Operation parameter %u has invalid type.\n"
                "    - Expected: %s\n"
                "    - Provided: %s\n"
                "    Hint: Ensure the operation parameters are of the correct type.\n",
                meta->mnemonic, i,
                mag_op_param_type_names[meta->op_param_layout[i].type],
                mag_op_param_type_names[params[i].type]
            );
            mag_print_separator(stderr);
            fputc('\n', stderr);
            fflush(stderr);
            return false;
        }
    }
    return true;
}

static bool mag_check_are_dtypes_compatible(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(a->dtype == b->dtype)) return true;
    mag_print_separator(stderr);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor dtypes are not compatible.\n"
        "    - Tensor 1 '%s' Datatype: %s\n"
        "    - Tensor 2 '%s' Datatype: %s\n"
        "    Hint: Ensure the input tensors have the same dtype.\n",
        meta->mnemonic,
        a->name, mag_dtype_meta_of(a->dtype)->name,
        b->name, mag_dtype_meta_of(b->dtype)->name
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Checks if the shape of a and b are equal. If not a detailed error message is printed.  Return true if valid, else false. */
static bool mag_check_is_shape_eq(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_shape_eq(a, b))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape_1, &a->shape, a->rank);
    mag_fmt_shape(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Tensor shapes must be equal.\n"
        "    - Tensor 1 '%s' Shape: %s\n"
        "    - Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transpose() or permute().\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Checks if the shape of a and b are broadcastable (b -> a). If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_shape_broadcastable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_can_broadcast(b, a))) return true;
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape_1, &a->shape, a->rank);
    mag_fmt_shape(&shape_2, &b->shape, b->rank);
    char bc[MAG_MAX_DIMS*2+3] = "[";
    int64_t pos = 1;
    int64_t mr = mag_xmax(a->rank, b->rank);
    for (int64_t d=0; d < mr; ++d) {
        int64_t asz = d < a->rank ? a->shape[a->rank-1-d] : 1;
        int64_t bsz = d < b->rank ? b->shape[b->rank-1-d] : 1;
        bc[pos++] = asz == bsz || asz == 1 || bsz == 1 ? 'Y' : 'N';
        bc[pos++] = d == mr-1 ? ']' : ',';
    }
    bc[pos] = '\0';
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be broadcast‑able (NumPy rules).\n"
        "    - Tensor 1 '%s' Shape: %s\n"
        "    - Tensor 2 '%s' Shape: %s\n"
        "    Broadcast‑ability per‑dim (right‑aligned): %s\n"
        "    Hint: Use unsqueeze()/view()/permute() to match shapes.\n",
        meta->mnemonic, a->name, shape_1, b->name, shape_2, bc);

    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Check if a and b can be matrix multiplied. If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_shape_matmulable(mag_op_t op, const mag_tensor_t* a, const mag_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(a->shape[1] == b->shape[0])) return true; /* Rows of a must match columns of b. */
    mag_print_separator(stderr);
    char shape_1[MAG_FMT_DIM_BUF_SIZE];
    char shape_2[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape_1, &a->shape, a->rank);
    mag_fmt_shape(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes are not compatible for matrix multiplication. The rows of the first tensor must match the columns of the second tensor.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transpose() or permute().\n",
        meta->mnemonic,
        a->name, shape_1,
        b->name, shape_2
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

/* Check if tensor is contiguous in memory. This is an required for some optimized compute algorithms. If not a detailed error message is printed. Return true if valid, else false. */
static bool mag_check_is_contiguous(mag_op_t op, const mag_tensor_t* a) {
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    if (mag_likely(mag_tensor_is_contiguous(a))) return true;
    mag_print_separator(stderr);
    char shape[MAG_FMT_DIM_BUF_SIZE];
    mag_fmt_shape(&shape, &a->shape, a->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Tensor '%s' must be contiguous. Shape: %s\n"
        "    Hint: Make tensor contiguous using contiguous().\n",
        meta->mnemonic,
        a->name,
        shape
    );
    mag_print_separator(stderr);
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

static bool mag_check_is_not_recording_gradients_for_inplace_op(mag_op_t op, mag_tensor_t* result, bool is_inplace) {
    if (mag_unlikely(is_inplace && result->ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER && result->flags & MAG_TFLAG_REQUIRES_GRAD)) {
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: In-place operation on tensor with gradient recording enabled.\n"
            "    Hint: Disable gradient recording temporarly for this op or use detach() to create a new detached tensor.\n",
            mag_op_meta_of(result->op)->mnemonic
        );
        return false;
    }
    return true;
}

/* Generic function which validates the tensors for common unary operations such as abs, neg, etc. */
static bool mag_validate_op_unary(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    bool ok = true;
    ok = ok && mag_check_is_not_recording_gradients_for_inplace_op(op, result, is_inplace);
    ok = ok && mag_check_is_shape_eq(op, result, inputs[0]);
    return ok;
}

/* Generic function which validates the tensors for common binary operations such as add, sub, etc. */
static bool mag_validate_op_binary(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    bool ok = true;
    ok = ok && mag_check_is_not_recording_gradients_for_inplace_op(op, result, is_inplace);
    ok = ok && mag_check_is_shape_eq(op, result, inputs[0]);
    ok = ok && mag_check_is_shape_broadcastable(op, inputs[0], inputs[1]);
    ok = ok && mag_check_is_contiguous(op, result);
    return ok;
}

/* Validation function for the transpose operation. */
static bool mag_validate_op_transpose(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return true;
}

/* Generic function which validates scalar operations such as adds, subs etc. */
static bool mag_validate_op_scalar(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    bool ok = true;
    ok = ok && mag_check_is_not_recording_gradients_for_inplace_op(op, result, is_inplace);
    ok = ok && mag_check_is_contiguous(op, inputs[0]);
    return ok;
}

/* Validation function for the matmul operation. */
static bool mag_validate_op_matmul(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_check_is_shape_matmulable(op, inputs[0], inputs[1]);
}

static bool mag_validate_op_repeat_rev(mag_op_t op, bool is_inplace, mag_tensor_t* result, mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_check_is_shape_broadcastable(op, inputs[0], inputs[1]);
}

static void mag_pre_validate_op(
    mag_op_t op,                /* Operator code */
    mag_tensor_t** inputs,      /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t numin,             /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_op_param_t* opps,      /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t numopps            /* Number of operation parameters. Must be same as specified in the op metadata. */
) {
    mag_assert2(op != MAG_OP_NOP);
    mag_assert(inputs && mag_check_are_inputs_valid(op, inputs, numin), "invalid input tensors for operation '%s'", mag_op_meta_of(op)->mnemonic);
    mag_assert(mag_check_are_op_params_valid(op, opps, numopps), "invalid parameters for operation '%s'", mag_op_meta_of(op)->mnemonic);
    if (numin > 1) { /* Check that dtype are compatible. */
        mag_assert(mag_check_are_dtypes_compatible(op, inputs[0], inputs[1]), "invalid dtypes for operation '%s'", mag_op_meta_of(op)->mnemonic);
    }
}


/*
** ###################################################################################################################
** Operator Result Constructors
** ###################################################################################################################
*/

static mag_tensor_t* mag_result_constructor_routine_isomorph(mag_tensor_t** inputs, const mag_op_param_t* params) {
    return mag_tensor_empty_like(*inputs);
}

static mag_tensor_t* mag_result_constructor_routine_view(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* base = *inputs;
    mag_tensor_t* result = mag_tensor_inplace_view(base);
    if (*base->name)
        mag_tensor_fmt_name(result, "%s (view)", base->name);
    return result;
}

static mag_tensor_t* mag_result_constructor_routine_scalar(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* base = *inputs;
    return mag_tensor_empty_scalar(base->ctx, base->dtype);
}

static mag_tensor_t* mag_result_constructor_routine_transposed(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_tensor_t* transposed = mag_result_constructor_routine_view(inputs, params);
    mag_swap(int64_t, transposed->shape[0], transposed->shape[1]);
    mag_swap(int64_t, transposed->strides[0], transposed->strides[1]);
    if (*inputs[0]->name)
        mag_tensor_fmt_name(transposed, "%s (T)", inputs[0]->name);
    return transposed;
}

static mag_tensor_t* mag_result_constructor_routine_permuted(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    mag_assert2(params != NULL);
    const mag_tensor_t* base = inputs[0];
    mag_tensor_t* permuted = mag_result_constructor_routine_view(inputs, params);
    int64_t axes[MAG_MAX_DIMS];
    for (int64_t i=0; i < base->rank; ++i)
        axes[i] = mag_op_param_unpack_u64_or_panic(params[i]);
    for (int64_t i=0; i < base->rank; ++i) {
        mag_assert2(axes[i] < base->rank);
        for (int64_t j=i+1; j < base->rank; ++j)
            mag_assert(axes[i] != axes[j], "Axes must be unique: %zu == %zu", axes[i], axes[j]);
    }
    int64_t tmp_shape[MAG_MAX_DIMS];
    int64_t tmp_stride[MAG_MAX_DIMS];
    memcpy(tmp_shape, base->shape, sizeof(tmp_shape));
    memcpy(tmp_stride, base->strides, sizeof(tmp_stride));
    for (int64_t i=0; i < base->rank; ++i) {
        permuted->shape[i] = tmp_shape[axes[i]];
        permuted->strides[i] = tmp_stride[axes[i]];
    }
    if (*base->name)
        mag_tensor_fmt_name(permuted, "%s (perm)", base->name);
    return permuted;
}

static mag_tensor_t* mag_result_constructor_routine_matmul(mag_tensor_t** inputs,  const mag_op_param_t* params) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[MAG_MAX_DIMS] = {0};
    int64_t* rd0 = shape;
    int64_t* rd1 = shape+1;
    int64_t rank = 0;
    if (inputs[0]->rank == 1 && inputs[1]->rank == 2) { /* (ℝⁿ)(ℝⁿˣʳ) → ℝʳ */
        *rd0 = 1;
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    } else if (inputs[0]->rank == 2 && inputs[1]->rank == 1) { /* (ℝᵐˣⁿ)(ℝⁿ) → ℝᵐ */
        *rd0 = inputs[0]->shape[0];
        rank = 1;
    } else if (inputs[0]->rank == 1 && inputs[1]->rank == 1) { /* (ℝⁿ)(ℝⁿ) → ℝ */
        rank = 1;
    } else { /* (ℝᵐˣⁿ)(ℝⁿˣʳ) → ℝᵐˣʳ */
        *rd0 = inputs[0]->shape[0];
        *rd1 = inputs[1]->shape[1];
        rank = 2;
    }
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, rank, shape, NULL, 0);
}

static mag_tensor_t* mag_result_constructor_routine_repeat_back(mag_tensor_t** inputs,  const mag_op_param_t* params) {
    return mag_tensor_init_internal(inputs[0]->ctx, inputs[0]->dtype, inputs[1]->rank, inputs[1]->shape, NULL, 0);
}

/*
** ###################################################################################################################
** Operator Backprop Impls
** ###################################################################################################################
*/

static void mag_op_backward_clone(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_view(mag_tensor_t* node, mag_tensor_t** grads) {
    *grads = mag_clone(node->grad);
}

static void mag_op_backward_transpose(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* t = mag_transpose(node->grad);
    *grads = mag_clone(t);
    mag_tensor_decref(t);
}

static void mag_op_backward_mean(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* scale = mag_tensor_full_like(x, (mag_e8m23_t)(1.0/(mag_e11m52_t)x->numel));
    *grads = mag_mul(scale, node->grad);
    mag_tensor_decref(scale);
}

static void mag_op_backward_sum(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* ones = mag_tensor_full_like(x, 1.0f);
    *grads = mag_mul(ones, node->grad);
    mag_tensor_decref(ones);
}

static void mag_op_backward_abs(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* step = mag_step(x);
    mag_tensor_t* one = mag_tensor_scalar(x->ctx, x->dtype, 1.0f);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* step2 = mag_mul(step, two);
    mag_tensor_t* sign = mag_sub(step2, one);
    grads[0] = mag_mul(node->grad, sign);
    mag_tensor_decref(two);
    mag_tensor_decref(one);
    mag_tensor_decref(step);
    mag_tensor_decref(step2);
    mag_tensor_decref(sign);
}

static void mag_op_backward_neg(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* m1 = mag_tensor_scalar(node->grad->ctx, node->grad->dtype, -1.f);
    grads[0] = mag_mul(node->grad, m1);
    mag_tensor_decref(m1);
}

static void mag_op_backward_log(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    grads[0] = mag_div(node->grad, x);
}

static void mag_op_backward_sqr(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* two_x = mag_mul(x, two);
    grads[0] = mag_mul(node->grad, two_x);
    mag_tensor_decref(two);
    mag_tensor_decref(two_x);
}

static void mag_op_backward_sqrt(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sqrt_x = mag_sqrt(x);
    mag_tensor_t* two = mag_tensor_scalar(x->ctx, x->dtype, 2.0f);
    mag_tensor_t* denom = mag_mul(sqrt_x, two);
    grads[0] = mag_div(node->grad, denom);
    mag_tensor_decref(two);
    mag_tensor_decref(sqrt_x);
    mag_tensor_decref(denom);
}

static void mag_op_backward_sin(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* cos_x = mag_cos(x);
    grads[0] = mag_mul(node->grad, cos_x);
    mag_tensor_decref(cos_x);
}

static void mag_op_backward_cos(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* sinx = mag_sin(x);
    mag_tensor_t* nsinx = mag_neg(sinx);
    grads[0] = mag_mul(node->grad, nsinx);
    mag_tensor_decref(sinx);
    mag_tensor_decref(nsinx);
}

static void mag_op_backward_exp(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* exp_x = mag_exp(x);
    grads[0] = mag_mul(node->grad, exp_x);
    mag_tensor_decref(exp_x);
}

static void mag_op_backward_softmax(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = mag_softmax(x);
    mag_tensor_t* tmp = mag_mul(node->grad, y);
    mag_tensor_t* sum_tmp = mag_sum(tmp, NULL, 0, false);
    mag_tensor_t* diff = mag_sub(node->grad, sum_tmp);
    grads[0] = mag_mul(y, diff);
    mag_tensor_decref(tmp);
    mag_tensor_decref(sum_tmp);
    mag_tensor_decref(diff);
    mag_tensor_decref(y);
}

static void mag_op_backward_sigmoid(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_sigmoid_dv(x);
    grads[0] = mag_mul(dv, node->grad);
    mag_tensor_decref(dv);
}

static void mag_op_backward_silu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_silu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_tanh(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_tanh_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_relu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* mask = mag_step(x);
    grads[0] = mag_mul(node->grad, mask);
    mag_tensor_decref(mask);
}

static void mag_op_backward_gelu(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* dv = mag_gelu_dv(x);
    grads[0] = mag_mul(node->grad, dv);
    mag_tensor_decref(dv);
}

static void mag_op_backward_add(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* grad = node->grad;
        if (!mag_tensor_is_shape_eq(x, y)) {
            grad = mag_repeat_back(grad, y);
        } else {
            grad = mag_clone(grad); /* Output gradients must be a new allocated tensor, so we clone. */
        }
        grads[1] = grad;
    }
}

static void mag_op_backward_sub(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_clone(node->grad);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* mg = mag_neg(node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmg = mg;
            mg = mag_repeat_back(pmg, y);
            mag_tensor_decref(pmg);
        }
        grads[1] = mg;
    }
}

static void mag_op_backward_mul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_mul(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* xg = mag_mul(x, node->grad);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pxg = xg;
            xg = mag_repeat_back(pxg, y);
            mag_tensor_decref(pxg);
        }
        grads[1] = xg;
    }
}

static void mag_op_backward_div(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        grads[0] = mag_div(node->grad, y);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* gx = mag_mul(node->grad, x);
        mag_tensor_t* yy = mag_mul(y, y);
        mag_tensor_t* gxyy = mag_div(gx, yy);
        mag_tensor_t* mgxyy = mag_neg(gxyy);
        if (!mag_tensor_is_shape_eq(x, y)) {
            mag_tensor_t* pmgxyy = mgxyy;
            mgxyy = mag_repeat_back(pmgxyy, y);
            mag_tensor_decref(pmgxyy);
        }
        grads[1] = mgxyy;
        mag_tensor_decref(gxyy);
        mag_tensor_decref(yy);
        mag_tensor_decref(gx);
    }
}

static void mag_op_backward_matmul(mag_tensor_t* node, mag_tensor_t** grads) {
    mag_tensor_t* x = node->op_inputs[0];
    mag_tensor_t* y = node->op_inputs[1];
    if (x->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* yt = mag_transpose(y);
        mag_tensor_t* ytc = mag_clone(yt);
        grads[0] = mag_matmul(node->grad, ytc);
        mag_tensor_decref(ytc);
        mag_tensor_decref(yt);
    }
    if (y->flags & MAG_TFLAG_REQUIRES_GRAD) {
        mag_tensor_t* xt = mag_transpose(x);
        mag_tensor_t* xtc = mag_clone(xt);
        grads[1] = mag_matmul(xtc, node->grad);
        mag_tensor_decref(xtc);
        mag_tensor_decref(xt);
    }
}

/*
** ###################################################################################################################
** Operator Metadata List
** ###################################################################################################################
*/

const mag_op_meta_t* mag_op_meta_of(mag_op_t opc) {
    static const mag_op_meta_t infos[MAG_OP__NUM] = {
        [MAG_OP_NOP] = {
            .mnemonic = "nop",
            .desc = "nop",
            .input_count = 0,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = NULL,
            .validator = NULL
        },
        [MAG_OP_CLONE] = {
            .mnemonic = "clone",
            .desc = "clone",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_clone,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_VIEW] = {
            .mnemonic = "view",
            .desc = "view",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_view,
            .r_alloc = &mag_result_constructor_routine_view,
            .validator = &mag_validate_op_unary
        },
        [MAG_OP_TRANSPOSE] = {
            .mnemonic = "transpose",
            .desc = "𝑥ᵀ",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_transpose,
            .r_alloc = &mag_result_constructor_routine_transposed,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_PERMUTE] = {
            .mnemonic = "permute",
            .desc = "permute",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_U64, .is_required=true}, /* perm axis : u32 */
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_permuted,
            .validator = &mag_validate_op_transpose
        },
        [MAG_OP_MEAN] = {
            .mnemonic = "mean",
            .desc = "(∑𝑥)∕𝑛",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_U64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_U64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_mean,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MIN] = {
            .mnemonic = "min",
            .desc = "min(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_U64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_U64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_MAX] = {
            .mnemonic = "max",
            .desc = "max(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_U64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_U64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_SUM] = {
            .mnemonic = "sum",
            .desc = "∑𝑥",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_U64, .is_required=true},  /* reduction dim count : u32 */
                {.type=MAG_OPP_U64, .is_required=true},  /* keepdim : bool */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
                {.type=MAG_OPP_U64, .is_required=false}, /* reduction dim : u32 [optional] */
            },
            .flags = MAG_OP_FLAG_NONE,
            .backward = &mag_op_backward_sum,
            .r_alloc = &mag_result_constructor_routine_scalar,
            .validator = &mag_validate_op_scalar
        },
        [MAG_OP_ABS] = {
            .mnemonic = "abs",
            .desc = "|𝑥|",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_abs,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SGN] = {
            .mnemonic = "sgn",
            .desc = "𝑥⁄",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_NEG] = {
            .mnemonic = "neg",
            .desc = "−𝑥",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_neg,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_LOG] = {
            .mnemonic = "log",
            .desc = "log₁₀(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_log,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQR] = {
            .mnemonic = "sqr",
            .desc = "𝑥²",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqr,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SQRT] = {
            .mnemonic = "sqrt",
            .desc = "√𝑥",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags =  MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sqrt,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIN] = {
            .mnemonic = "sin",
            .desc = "sin(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sin,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_COS] = {
            .mnemonic = "cos",
            .desc = "cos(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_cos,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_STEP] = {
            .mnemonic = "step",
            .desc = "𝐻(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_EXP] = {
            .mnemonic = "exp",
            .desc = "𝑒ˣ",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_exp,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_FLOOR] = {
            .mnemonic = "floor",
            .desc = "⌊𝑥⌋",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_CEIL] = {
            .mnemonic = "ceil",
            .desc = "⌈𝑥⌉",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ROUND] = {
            .mnemonic = "round",
            .desc = "⟦𝑥⟧",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX] = {
            .mnemonic = "softmax",
            .desc = "𝑒ˣⁱ∕∑𝑒ˣʲ",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_softmax,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SOFTMAX_DV] = {
            .mnemonic = "softmax_dv",
            .desc = "𝑑⁄𝑑𝑥 softmax(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID] = {
            .mnemonic = "sigmoid",
            .desc = "1∕(1 + 𝑒⁻ˣ)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = mag_op_backward_sigmoid,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SIGMOID_DV] = {
            .mnemonic = "sigmoid_dv",
            .desc = "𝑑⁄𝑑𝑥 sigmoid(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_HARD_SIGMOID] = {
            .mnemonic = "hard_sigmoid",
            .desc = "max(0,min(1,0.2×𝑥+0.5))",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU] = {
            .mnemonic = "silu",
            .desc = "𝑥∕(1+𝑒⁻ˣ)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_silu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SILU_DV] = {
            .mnemonic = "silu_dv",
            .desc = "𝑑⁄𝑑𝑥 silu(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH] = {
            .mnemonic = "tanh",
            .desc = "tanh(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_tanh,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_TANH_DV] = {
            .mnemonic = "tanh_dv",
            .desc = "𝑑⁄𝑑𝑥 tanh(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU] = {
            .mnemonic = "relu",
            .desc = "max(0, 𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_relu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_RELU_DV] = {
            .mnemonic = "relu_dv",
            .desc = "𝑑⁄𝑑𝑥 relu(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU] = {
            .mnemonic = "gelu",
            .desc = "0.5×𝑥×(1+erf(𝑥∕√2))",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_gelu,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_GELU_DV] = {
            .mnemonic = "gelu_dv",
            .desc = "𝑑⁄𝑑𝑥 gelu(𝑥)",
            .input_count = 1,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_unary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_ADD] = {
            .mnemonic = "add",
            .desc = "𝑥 + 𝑦",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_add,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_SUB] = {
            .mnemonic = "sub",
            .desc = "𝑥 − 𝑦",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_sub,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_MUL] = {
            .mnemonic = "mul",
            .desc = "𝑥 ⊙ 𝑦",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_mul,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_DIV] = {
            .mnemonic = "div",
            .desc = "𝑥 ∕ 𝑦",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE | MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_div,
            .r_alloc = &mag_result_constructor_routine_isomorph,
            .validator = &mag_validate_op_binary,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 250000
            }
        },
        [MAG_OP_MATMUL] = {
            .mnemonic = "matmul",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORT_CPU_MULTITHREADING,
            .backward = &mag_op_backward_matmul,
            .r_alloc = &mag_result_constructor_routine_matmul,
            .validator = &mag_validate_op_matmul,
            .cpu = {
                .thread_growth = 0.1,
                .thread_treshold = 10000
            }
        },
        [MAG_OP_REPEAT_BACK] = {
            .mnemonic = "repeat_rev",
            .input_count = 2,
            .op_param_layout = {
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
                {.type=MAG_OPP_NONE, .is_required=false},
            },
            .flags = MAG_OP_FLAG_SUPPORTS_INPLACE,
            .backward = NULL,
            .r_alloc = &mag_result_constructor_routine_repeat_back,
            .validator = mag_validate_op_repeat_rev
        }
    };
    return infos+opc;
}

/* Execute init/normal operator on R. */
static void MAG_HOTPROC mag_op_exec(mag_tensor_t* R, mag_compute_device_t* dvc, mag_exec_stage_t stage) {
    void (*exec)(mag_compute_device_t*, mag_tensor_t*) = stage == MAG_STAGE_INIT ? dvc->eager_exec_init : dvc->eager_exec_fwd;
    (*exec)(dvc, R); /* Dispatch to backend. */
}

extern void mag_tensor_detach_inplace(mag_tensor_t* target);

/* Execute an operator on the active compute device and return result tensor. */
static mag_tensor_t* MAG_HOTPROC mag_tensor_operator(
    mag_ctx_t* ctx,             /* Context to use. All involved tensors must be allocated from this context. */
    mag_op_t op,                /* Operator code */
    bool inplace,               /* Attempt to perform inplace operation? e.g. r <- x += y instead of      r <- x + y */
    mag_tensor_t** inputs,      /* Input tensors. Must point to an array of 'num_inputs' (N) non-null tensors. */
    uint32_t num_inputs,        /* Number of valid non-null input tensors in the inputs array. Must be same as specified in the op metadata. */
    const mag_op_param_t* params,    /* Operation parameters or NULL. Must be same as specified in the op metadata. */
    uint32_t num_params,        /* Number of operation parameters. Must be same as specified in the op metadata. */
    mag_exec_stage_t stage      /* Graph evaluation direction. */
) {
    mag_pre_validate_op(op, inputs, num_inputs, params, num_params); /* Validate that function inputs are correct. */

    /* Query validate and result constructor functions for the scheduled opcode. */
    const mag_op_meta_t* meta = mag_op_meta_of(op);
    mag_tensor_t* (*r_alloc)(mag_tensor_t**, const mag_op_param_t*) = meta->r_alloc;                         /* Get result allocator function. */
    bool (*validate_op)(mag_op_t, bool, mag_tensor_t*, mag_tensor_t**, const mag_op_param_t*) = meta->validator;   /* Get validator function. */
    inplace &= !!(meta->flags & MAG_OP_FLAG_SUPPORTS_INPLACE);                                          /* Inplace operation requested and supported? */
    mag_tensor_t* result = inplace ? mag_tensor_inplace_view(*inputs) : (*r_alloc)(inputs, params);     /* If inplace, result views x (input 0), else a new result tensor is allocated. */
    result->op = op;                                                                                    /* Set opcode of result. */
    mag_assert((*validate_op)(op, inplace, result, inputs, params), "Invalid operation %s.", meta->mnemonic);    /* Now validate the full operation and panic if something doesn't work out. */
    mag_assert2(num_inputs <= MAG_MAX_OP_INPUTS);                                                       /* Assert correct input tensor count. */
    bool is_recording_grads = !!(ctx->flags & MAG_CTX_FLAG_GRAD_RECORDER);
    for (uint32_t i=0; i < num_inputs; ++i) { /* Apply input tensor's gradient rules and increase their lifetime. */
        mag_tensor_t* input = inputs[i];
        result->op_inputs[i] = input;
        /* If gradient tracking is enabled, the result tensor inherits the input's gradient rules. */
        if (is_recording_grads) {
            result->flags |= input->flags & MAG_TFLAG_REQUIRES_GRAD; /* Set gradient tracking flag if set in input. */
            mag_tensor_incref(input);                                /* Keep input alive for the backward pass. */
        }
    }
    if (params) /* If available, copy operation parameters to result */
        memcpy(result->op_params, params, num_params*sizeof(*params));
    mag_op_exec(result, ctx->device, stage);  /* Now execute the operator. */
    if (!is_recording_grads)
        mag_tensor_detach_inplace(result); /* If gradient are not recorded, detach the tensor's parents (clear parent and opcode). TODO: why are we doing this? */
    return result;
}

mag_tensor_t* mag_clone(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CLONE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_view(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_VIEW, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_transpose(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TRANSPOSE, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_permute(mag_tensor_t* x, const int64_t* _Nonnull dims, uint32_t num_dims) {
    mag_assert(dims != NULL, "Invalid permutation dimensions");
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid permutation dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    for (uint32_t i=0; i < num_dims; ++i)
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_PERMUTE, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mean(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MEAN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_min(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_max(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sum(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_SUM, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_argmin(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MIN, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_argmax(mag_tensor_t* x, const int64_t* dims, uint32_t num_dims, bool keepdim) {
    mag_assert(num_dims <= MAG_MAX_DIMS, "Invalid reduction dimensions count, max %d but is %u", MAG_MAX_DIMS, num_dims);
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(num_dims)); /* Store number of reduction axes in op_params[0] */
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(!!keepdim)); /* Store keepdim in op_params[1] */
    for (uint32_t i=2; i < num_dims; ++i) /* Store reduction axes in op_params[2:] */
        mag_op_param_layout_insert(&layout, mag_op_param_wrap_u64(dims[i]));
    return mag_tensor_operator(x->ctx, MAG_OP_MAX, false, &x, 1, layout.slots, layout.count, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_abs_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ABS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sgn_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SGN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_neg_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_NEG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_log_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_LOG, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqr_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sqrt_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SQRT, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sin_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIN, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_cos_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_COS, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_step_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_STEP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_exp_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_EXP, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_floor_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_FLOOR, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_ceil_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_CEIL, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_round_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_ROUND, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_softmax_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SOFTMAX_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sigmoid_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SIGMOID_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_hard_sigmoid_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_HARD_SIGMOID, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_silu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_SILU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_tanh_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_TANH_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_relu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_RELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, false, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_gelu_dv_(mag_tensor_t* x) {
    return mag_tensor_operator(x->ctx, MAG_OP_GELU_DV, true, &x, 1, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_add_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_ADD, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_sub_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_SUB, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_mul_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MUL, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_div_(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_DIV, true, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_adds(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_adds_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_ADDS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_subs(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_subs_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_SUBS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_muls(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_muls_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_MULS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_divs(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_divs_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_DIVS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_pows(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_POWS, false, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_pows_(mag_tensor_t* x, mag_e8m23_t xi) {
    mag_op_param_t param = mag_op_param_wrap_e8m23(xi);
    return mag_tensor_operator(x->ctx, MAG_OP_POWS, true, &x, 1, &param, 1, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_matmul(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_MATMUL, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

mag_tensor_t* mag_repeat_back(mag_tensor_t* x, mag_tensor_t* y) {
    return mag_tensor_operator(x->ctx, MAG_OP_REPEAT_BACK, false, (mag_tensor_t*[]){x, y}, 2, NULL, 0, MAG_STAGE_EVAL);
}

void mag_tensor_fill_from_floats(mag_tensor_t* t, const mag_e8m23_t* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CVT_E8M23, 0, (void*)data, len*sizeof(*data));
}

void mag_tensor_fill_from_raw_bytes(mag_tensor_t* t, const void* data, size_t len) {
    mag_assert(data && len, "invalid data pointer or length");
    mag_storage_buffer_t* sto = t->storage;
    (*sto->transfer)(sto, MAG_TRANSFER_DIR_H2D, MAG_TRANSFER_OP_CPY, 0, (void*)data, len);
}

void mag_tensor_fill(mag_tensor_t* t, mag_e8m23_t x) {
    t->init_op = MAG_IOP_BROADCAST;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(x));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_uniform(mag_tensor_t* t, mag_e8m23_t min, mag_e8m23_t max) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    t->init_op = MAG_IOP_RAND_UNIFORM;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(min));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(max));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}

void mag_tensor_fill_random_normal(mag_tensor_t* t, mag_e8m23_t mean, mag_e8m23_t stddev) {
    mag_assert2(t->ctx->device_type == MAG_COMPUTE_DEVICE_TYPE_CPU);
    t->init_op = MAG_IOP_RAND_NORMAL;
    mag_op_param_layout_t layout;
    mag_op_param_layout_init(&layout);
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(mean));
    mag_op_param_layout_insert(&layout, mag_op_param_wrap_e8m23(stddev));
    mag_op_param_layout_transfer(&layout, &t->init_op_params);
    mag_op_exec(t, t->ctx->device, MAG_STAGE_INIT);
}
