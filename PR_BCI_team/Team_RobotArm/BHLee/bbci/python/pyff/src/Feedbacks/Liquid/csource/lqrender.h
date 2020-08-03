#ifndef __LQRENDER__
#define __LQRENDER__
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384338
#endif
#define sqr(X) ((X)*(X))
#define rand_double() ((double)rand() / (double)RAND_MAX)
#define distance(X1, Y1, X2, Y2) (sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)))

#define MAX_PTS 200000
#define MAX_EDGES 32
#define MAX_CLASSES 2048
//Cache values




//max height of a single element
#define HEIGHT_SCALE 8e6

#define min(x,y) (((x)>(y)) ? (y) : (x))

// A single lookup table (e.g. for one type of fn)
typedef struct lq_table {
int *table;
int width, height;
int falloff_distance;
} lq_table;

// A single point (encapsulating position and a fn and list ptrs)
typedef struct lq_lut {
lq_table *table;
int x_offset,  y_offset;
struct lq_lut *prev, *next;
int blob_class; // which "blob" are we part of
} lq_lut;


//An index of a section of points (delimiting one complete blob polygon)
typedef struct lq_section
{
    int start;
    int end;
} lq_section;

typedef struct lq_edge {
int x, y, dir;
struct lq_edge *next;
} lq_edge;

// A collection of points and cache structures
typedef struct lq_data {
int width, height;
int **y_edges;
int *n_y_edges;
char *edge_cache;
char *dir_cache;
char *class_cache;
lq_section *blob_sections;
int n_sections;
int *x_list, *y_list;
int n_list_pts;
lq_lut *lut_list;
int top, left, bottom, right;
int *boundary;
} lq_data;



#define PIXEL_UNDER 1
#define PIXEL_OVER 2
#define PIXEL_EDGE 4


lq_table *gen_table(int width, int height);
void set_gaussian_table(lq_table *table, int std);
void set_cauchy_table(lq_table *table, int std);
void lut_set_pos(lq_lut *lut, int x, int y);
void lut_get_pos(lq_lut *lut, int *x, int *y);
lq_lut *new_lut(int x, int y, lq_table *table);
lq_data *new_data(int width, int height, int top, int bottom, int left, int right);
void data_add_lut(lq_data *data, lq_lut *lut);
int data_remove_lut(lq_data *data, lq_lut *lut);
int get_height(lq_data *data, int x, int y);
void data_set_boundary(lq_data *data, int *boundary);
int trace_pixel(lq_data *data, int x, int y, int thresh);
void add_edge_point(lq_data *data, int x, int y, int dir);
void trace(lq_data *data, lq_lut *lut, int thresh);
void data_render(lq_data *data, int thresh);
int compute_direction(lq_data *data, int x, int y, int thresh, int *idx);
#endif