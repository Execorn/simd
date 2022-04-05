#include "TXLib.h"
#include "immintrin.h"

#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")

/* 
* Constants for the (0, 0) point and window parameters
*/
const float O_x = -1.325f,
            O_y = 0;


const int   WINDOW_WIDTH  = 800;
const int   WINDOW_HEIGHT = 600;


int main() {
    txCreateWindow (WINDOW_WIDTH, WINDOW_HEIGHT);
    Win32::_fpreset();
    txBegin();

    typedef RGBQUAD (&scr_t) [WINDOW_HEIGHT][WINDOW_WIDTH];
    scr_t scr = (scr_t) *txVideoMemory();
    
    const int    max_points  = 256;
    const float  dx    = 1/800.f, dy = 1/800.f;

    const __m256 maxR_vector = _mm256_set1_ps (100.f);
    const __m256 _255  = _mm256_set1_ps (255.f);
    const __m256 _76543210 = _mm256_set_ps  (7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f, 0.f);
    const __m256 maxP_vector  = _mm256_set1_ps (max_points);

    float xC = 0.f, yC = 0.f, scale = 1.f;

    uint8_t do_draw_cycle = 1;
    while (do_draw_cycle) {
        if (GetAsyncKeyState (VK_ESCAPE)) do_draw_cycle = 0;
        
        if (txGetAsyncKeyState (VK_RIGHT)) xC    += dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_LEFT))  xC    -= dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_DOWN))  yC    -= dy * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_UP))    yC    += dy * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState ('A'))      scale += dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState ('Z'))      scale -= dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);

        #pragma omp parallel for
        for (int iy = 0; iy < 1080 * do_draw_cycle; iy++) {
            if (GetAsyncKeyState (VK_ESCAPE)) do_draw_cycle = 0;

            float y0 = ( ((float) iy - (float) WINDOW_HEIGHT / 2) * dy) * scale + O_y + yC;

            #pragma omp parallel for
            for (int ix = 0; ix < WINDOW_WIDTH; ix += 8/*, x0 += dx * 8 * scale*/) { 
                float x0 = ( ((float) ix - (float) WINDOW_WIDTH / 2) * dx ) * scale + O_x + xC;
                __m256 X0 = _mm256_add_ps (_mm256_set1_ps (x0), _mm256_mul_ps (_76543210, _mm256_set1_ps (dx * scale)));
                __m256 Y0 =             _mm256_set1_ps (y0);

                __m256 X = X0, Y = Y0;
                
                __m256i N = _mm256_setzero_si256();
                
                uint8_t do_calc_points = 1;
                for (int n = 0; n < max_points; n++) {
                    if (do_calc_points) {
                        __m256 x2 = _mm256_mul_ps (X, X),
                           y2 = _mm256_mul_ps (Y, Y);
                           
                        __m256 r2 = _mm256_add_ps (x2, y2);

                        __m256 cmp = _mm256_cmp_ps (r2, maxR_vector, _CMP_LE_OQ);
                        int mask   = _mm256_movemask_ps (cmp);
                        if (!mask) do_calc_points = 0;
                            
                        N = _mm256_sub_epi32 (N, _mm256_castps_si256 (cmp));

                        __m256 xy = _mm256_mul_ps (X, Y);

                        X = _mm256_add_ps (_mm256_sub_ps (x2, y2), X0);
                        Y = _mm256_add_ps (_mm256_add_ps (xy, xy), Y0);
                    }
                }
                
                /*
                * Filter mask for better point drawing
                */
                __m256 I = _mm256_mul_ps (_mm256_sqrt_ps (_mm256_sqrt_ps (_mm256_div_ps (_mm256_cvtepi32_ps (N), maxP_vector))), _255);

                for (int i = 0; i < 8; i++) {
                    int*   pn = (int*)   &N;
                    float* pI = (float*) &I;

                    BYTE    c     = (BYTE) pI[i];
                    RGBQUAD color = (pn[i] < max_points)? RGBQUAD { (BYTE) (255-c), (BYTE) (c%2 * 64), c } : RGBQUAD {};
 
                    scr[iy][ix+i] = color;
                }
            }
        }
            
    printf ("\t\r%.0lf", txGetFPS());
    txUpdateWindow();  
    }

    txDisableAutoPause();
}
