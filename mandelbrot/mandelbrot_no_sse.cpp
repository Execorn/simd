
#include "TXLib.h"

//=================================================================================================

const float ROI_X = -1.325f,
            ROI_Y = 0;

//=================================================================================================

#pragma GCC diagnostic ignored "-Wfloat-equal"

inline void mm_set_ps   (float mm[4], float val0, float val1, float val2, float val3)
    {
    mm[0] = val0; mm[1] = val1; mm[2] = val2; mm[3] = val3;
    }
    
inline void mm_set_ps1   (float mm[4],  float val)                              { for (int i = 0; i < 4; i++) mm[i] = val;                      }
inline void mm_cpy_ps    (float mm[4],  const float mm2[4])                     { for (int i = 0; i < 4; i++) mm[i] = mm2[i];                   }
                                                                                                                                              
inline void mm_add_ps    (float mm[4],  const float mm1[4], const float mm2[4]) { for (int i = 0; i < 4; i++) mm[i] = mm1[i] + mm2[i];          }
inline void mm_sub_ps    (float mm[4],  const float mm1[4], const float mm2[4]) { for (int i = 0; i < 4; i++) mm[i] = mm1[i] - mm2[i];          }
inline void mm_mul_ps    (float mm[4],  const float mm1[4], const float mm2[4]) { for (int i = 0; i < 4; i++) mm[i] = mm1[i] * mm2[i];          }

inline void mm_add_epi32 (int   mm[4],  const int   mm1[4], const float mm2[4]) { for (int i = 0; i < 4; i++) mm[i] = mm1[i] + (int) mm2[i];    }

inline void mm_cmple_ps  (float cmp[4], const float mm1[4], const float mm2[4]) { for (int i = 0; i < 4; i++) if (mm1[i] <= mm2[i]) cmp[i] = 1; }
           
inline int  mm_movemask_ps (const float cmp[4])
    {
    int mask = 0;
    for (int i = 0; i < 4; i++) mask |= (!!cmp[i] << i);
    return mask;
    }

//-------------------------------------------------------------------------------------------------

int main()
    {
    txCreateWindow (800, 600);
    Win32::_fpreset();
    txBegin();

    typedef RGBQUAD (&scr_t) [600][800];
    scr_t scr = (scr_t) *txVideoMemory();
    
    const int nMax = 256;
    const float dx = 1/800.f, dy = 1/800.f;
    float r2Max[4] = {}; mm_set_ps1 (r2Max, 100.f);
    float _3210[4] = {}; mm_set_ps  (_3210, 0.f, 1.f, 2.f, 3.f);
    
    float xC = 0.f, yC = 0.f, scale = 1.f;

    for (;;)
        {
        if (GetAsyncKeyState (VK_ESCAPE)) break;
        
        if (txGetAsyncKeyState (VK_RIGHT)) xC    += dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_LEFT))  xC    -= dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_DOWN))  yC    += dy * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState (VK_UP))    yC    -= dy * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState ('A'))      scale += dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);
        if (txGetAsyncKeyState ('Z'))      scale -= dx * (txGetAsyncKeyState (VK_SHIFT)? 100.f : 10.f);

        for (int iy = 0; iy < 600; iy++) 
            {
            if (GetAsyncKeyState (VK_ESCAPE)) break;

            float x0 = ( (          - 400.f) * dx + ROI_X + xC ) * scale,
                  y0 = ( ((float)iy - 300.f) * dy + ROI_Y + yC ) * scale;
                  
            for (int ix = 0; ix < 800; ix += 4, x0 += dx*4)
                {
                float DX[4] = {}; mm_set_ps1 (DX, dx); mm_mul_ps (DX, DX, _3210); 

                float X0[4] = {}; mm_set_ps1 (X0, x0); mm_add_ps (X0, X0, DX);
                float Y0[4] = {}; mm_set_ps1 (Y0, y0);

                float X [4] = {}; mm_cpy_ps (X, X0);
                float Y [4] = {}; mm_cpy_ps (Y, Y0);
                      
                int   N [4] = {0, 0, 0, 0};

                for (int n = 0; n < nMax; n++)
                    {
                    float x2[4] = {}; mm_mul_ps (x2, X, X);
                    float y2[4] = {}; mm_mul_ps (y2, Y, Y);

                    float r2[4] = {}; mm_add_ps (r2, x2, y2);
                    
                    float cmp[4]= {}; mm_cmple_ps (cmp, r2, r2Max);

                    int mask = mm_movemask_ps (cmp);
                    if (!mask) break;
                    
                    mm_add_epi32 (N, N, cmp);

                    float xy[4] = {}; mm_mul_ps (xy, X, Y);
                    
                    mm_sub_ps (X, x2, y2); mm_add_ps (X, X, X0);
                    mm_add_ps (Y, xy, xy); mm_add_ps (Y, Y, Y0);
                    }
                   
                for (int i = 0; i < 4; i++)
                    {
                    float I = sqrtf (sqrtf ((float)N[i] / (float)nMax)) * 255.f;
                    
                    BYTE    c     = (BYTE) I;
                    RGBQUAD color = (N[i] < nMax)? RGBQUAD { (BYTE) (255-c), (BYTE) (c%2 * 64), c } : RGBQUAD {};
                    
                    scr[iy][ix+i] = color;
                    }
                }
            }
            
        printf ("\t\r%.0lf", txGetFPS());
        txUpdateWindow();
        }

    txDisableAutoPause();
    }
