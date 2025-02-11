/* (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include <magnetron.h>

#include <stdio.h>
#include <time.h>

int main(void) {
    printf("Running sanity test...\n");
    mag_set_log_mode(true); /* Enable logging */

    mag_ctx_t* ctx = mag_ctx_create(MAG_COMPUTE_DEVICE_TYPE_CPU);

    mag_tensor_t* a = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 4096, 4096, 48);
    mag_tensor_fill(a, 0.0f);

    mag_tensor_t* b = mag_tensor_create_3d(ctx, MAG_DTYPE_F32, 4096, 4096, 48);
    mag_tensor_fill(b, 0.0f);

    printf("Computing...\n");
    clock_t begin = clock();
    mag_tensor_t* result = mag_add(a, b); /* Compute result = a + b */
    clock_t end = clock();
    double secs = (double)(end - begin)/CLOCKS_PER_SEC;
    printf("Computed in %f s\n", secs);

    /* Free tensors */
    mag_tensor_decref(result);
    mag_tensor_decref(b);
    mag_tensor_decref(a);

    /* Free context */
    mag_ctx_destroy(ctx);

    printf("Done!\n");

    return 0;
}
