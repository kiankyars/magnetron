/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include <magnetron.h>

#include <stdio.h>

/*
** Simple sanity test: I
** Initialize Magnetron and compute matrix product of two random 1024x1024 matrices.
*/
int main(void) {
    printf("Running sanity test...\n");
    mag_set_log_mode(true); /* Enable logging */

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    printf("Computing...\n");

    mag_tensor_t* a = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 4096, 4096);
    mag_tensor_fill_random_uniform(a, -1.0f, 1.0f);

    mag_tensor_t* b = mag_tensor_create_2d(ctx, MAG_DTYPE_F32, 4096, 4096);
    mag_tensor_fill_random_uniform(b, -1.0f, 1.0f);

    mag_tensor_t* result = mag_matmul(a, b); /* Compute result = a @ b */
    printf("Computed!\n");

    /* Free tensors */
    mag_tensor_decref(result);
    mag_tensor_decref(b);
    mag_tensor_decref(a);

    /* Free context */
    mag_ctx_destroy(ctx);

    printf("Done!\n");

    return 0;
}
