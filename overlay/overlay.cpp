
#include "TXLib.h"
#include <immintrin.h>


#pragma GCC optimize("Ofast")
#pragma GCC target("avx,avx2,fma")

//-------------------------------------------------------------------------------------------------

typedef RGBQUAD (&scr_t) [600][800];

inline scr_t LoadImage (const char* filename) {
    RGBQUAD* mem = NULL;
    HDC dc = txCreateDIBSection (800, 600, &mem);
    txBitBlt (dc, 0, 0, 0, 0, dc, 0, 0, BLACKNESS);

    HDC image = txLoadImage (filename);
    txBitBlt (dc, (txGetExtentX (dc) - txGetExtentX (image)) / 2, 
                  (txGetExtentY (dc) - txGetExtentY (image)) / 2, 0, 0, image);
    txDeleteDC (image);

    return (scr_t) *mem;
    }

//----------------------------------------------------------------
// < Constants for 0/1 bytes and vectors

const char units_only       = 255u,
           unit_and_zeros   = 0x80u;
           
// ! Setting with 0-bytes
const __m128i only_zeros =                    _mm_set_epi8 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); 
// ? Setting with 1-bits
const __m128i only_units = _mm_cvtepu8_epi16 ( _mm_set_epi8(units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only,
                                                            units_only, units_only));
//----------------------------------------------------------------

int main() {
    // ! Creating window of size 800x600, WITHOUT DRAWING IT YET
    txCreateWindow (800, 600); 
    Win32::_fpreset();
    txBegin();


    //----------------------------------------------------------------
    // < Setting images, getting image video memory
    scr_t foreground  = (scr_t) LoadImage ("images/AskhatCat.bmp");
    scr_t background  = (scr_t) LoadImage ("images/Table.bmp");
    scr_t scr   = (scr_t) *txVideoMemory();
    //----------------------------------------------------------------


    for (int img = 0; ; ++img) {
        if (GetAsyncKeyState (VK_ESCAPE)) {
            break;
        }

        if (!GetKeyState (VK_SPACE)) {
            #pragma omp parallel for
            for (int y = 0; y < 600; ++y) {
                #pragma omp parallel for
                for (int x = 0; x < 800; x += 4) {
                    // ! Reading current background pixel, current foreground pixel, 
                    // ? frontal_byte = foreground[y][x]
                    __m128i frontal_byte    = _mm_load_si128 ((__m128i*) &foreground[y][x]); 
                    __m128i background_byte = _mm_load_si128 ((__m128i*) &background [y][x]);


                    // ! Putting 2 upper value of image(s) byte to 2 lower values of zero-vector, saving 
                    // ? shifting frontal byte 64 times, so we get '0000000.....00000(frontal_byte)'
                    __m128i frontal_shifted     = (__m128i) _mm_movehl_ps ((__m128) only_zeros, (__m128)    frontal_byte);   
                    __m128i backfround_shifted  = (__m128i) _mm_movehl_ps ((__m128) only_zeros, (__m128) background_byte);


                    // ! Zero-extend frontal_bytes 
                    // ? frontal_byte[i] = (BYTE) frontal_byte[i]
                    frontal_byte    = _mm_cvtepu8_epi16 (frontal_byte);                                             
                    frontal_shifted = _mm_cvtepu8_epi16 (frontal_shifted);


                    // ! Zero-extend background_bytes 
                    // ? background_byte[i] = (BYTE) background_byte[i]
                    background_byte     = _mm_cvtepu8_epi16 (background_byte);
                    backfround_shifted  = _mm_cvtepu8_epi16 (backfround_shifted);


                    // ! FIRST MASK TO SHUFFLE
                    // ? shuffling like that: frontal_shuffled [for r0/b0/b0...] = a0...
                    static const __m128i shuffle_control_mask = _mm_set_epi8 (  unit_and_zeros, 14, 
                                                                                unit_and_zeros, 14, 
                                                                                unit_and_zeros, 14, 
                                                                                unit_and_zeros, 14,
                                                                                unit_and_zeros, 6,  
                                                                                unit_and_zeros, 6,  
                                                                                unit_and_zeros, 6,  
                                                                                unit_and_zeros, 6);
                    
                    // < actually shuffling by bytes
                    __m128i frontal_shuffled         = _mm_shuffle_epi8 (frontal_byte,    shuffle_control_mask);     
                    __m128i frontal_shifted_shuffled = _mm_shuffle_epi8 (frontal_shifted, shuffle_control_mask);

                    // ! calculating alpha-value (opacity) of new image
                    // ? frontal_byte *= frontal_shuffled    
                    frontal_byte    = _mm_mullo_epi16 (frontal_byte, frontal_shuffled);                           
                    frontal_shifted = _mm_mullo_epi16 (frontal_shifted, frontal_shifted_shuffled);

                    // ! now getting background byte opacity with multiplication
                    // ? background_byte *= (255-frontal_shuffled)
                    background_byte     = _mm_mullo_epi16 (background_byte,     _mm_sub_epi16 (only_units, frontal_shuffled));                                 
                    backfround_shifted  = _mm_mullo_epi16 (backfround_shifted,  _mm_sub_epi16 (only_units, frontal_shifted_shuffled));

                    __m128i sum = _mm_add_epi16 (frontal_byte, background_byte);                                       // sum = frontal_byte*frontal_shuffled + background_byte*(255-frontal_shuffled)
                    __m128i SUM = _mm_add_epi16 (frontal_shifted, backfround_shifted);

                    static const __m128i shuffle_control_sum = _mm_set_epi8 (unit_and_zeros,  
                                                                unit_and_zeros,  
                                                                unit_and_zeros,  
                                                                unit_and_zeros, 
                                                                unit_and_zeros, 
                                                                unit_and_zeros, 
                                                                unit_and_zeros, 
                                                                unit_and_zeros, 
                                                                 15, 13, 11, 9, 7, 5, 3, 1);
                    sum = _mm_shuffle_epi8 (sum, shuffle_control_sum);                                      // sum[i] = (sium[i] >> 8) = (sum[i] / 256)
                    SUM = _mm_shuffle_epi8 (SUM, shuffle_control_sum);
                
                    __m128i color = (__m128i) _mm_movelh_ps ((__m128) sum, (__m128) SUM);  // color = (sumHi << 8*8) | sum

                    _mm_store_si128 ((__m128i*) &scr[y][x], color);
                }
            }    
        } else {
            for (int y = 0; y < 600; ++y) {
                for (int x = 0; x < 800; ++x) {
                    RGBQUAD* frontal_byte = &foreground[y][x];
                    RGBQUAD* background_byte = &background [y][x];
                    
                    uint16_t frontal_shuffled  = frontal_byte->rgbReserved;

                    scr[y][x]   = { (BYTE) ( (frontal_byte->rgbBlue  * (frontal_shuffled) + background_byte->rgbBlue  * (255-frontal_shuffled)) >> 8 ),
                                    (BYTE) ( (frontal_byte->rgbGreen * (frontal_shuffled) + background_byte->rgbGreen * (255-frontal_shuffled)) >> 8 ),
                                    (BYTE) ( (frontal_byte->rgbRed   * (frontal_shuffled) + background_byte->rgbRed   * (255-frontal_shuffled)) >> 8 ) };
                }
            }        
        }
                
        if (img % 10 == 0) {
            printf ("\t\r%.0lf", txGetFPS() * 10);
        }    
        txUpdateWindow();
    }
    txDisableAutoPause();
}
    
