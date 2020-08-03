#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lqrender.h"

#define DIR_EDGE -1

#define DIR_LEFT 0
#define DIR_UP 1
#define DIR_RIGHT 2
#define DIR_DOWN 3
#define DIR_BAD 4


//The walking table in bit order: 1 2
//                                                    3 4
// 0 =left, 1=up, 2=right, 3=down, 4=invalid
int walking_table[] = {
DIR_RIGHT, DIR_UP, DIR_RIGHT, DIR_RIGHT,
DIR_LEFT,  DIR_UP, DIR_LEFT,  DIR_RIGHT,
DIR_DOWN,  DIR_UP, DIR_DOWN,  DIR_DOWN,
DIR_LEFT,  DIR_UP, DIR_LEFT,  DIR_BAD
};

lq_table *gen_table(int width, int height)
{
lq_table *table;

table = calloc(sizeof(*table), 1);
table->width = width;
table->height = height;
table->table = calloc(sizeof(*(table->table)), width*height);

//Distance at which it is too far to bother to continue checking
table->falloff_distance = min(width/2-1, height/2-1);
return table;
}


void set_gaussian_table(lq_table *table, int std)
{
int i, j, x, y;
double d;
double norm_max;
int width, height;
width = table->width;
height = table->height;
//scale maximum height...
norm_max = exp(0);
//set min
table->falloff_distance = min(table->falloff_distance, std*4);
for(i=0;i<width;i++)
    for(j=0;j<height;j++)
    {
        x = i - width/2;
        y = j - height/2;
        d = x*x + y*y;
        //Heights are scaled to max 2^23ish, so up to 512 particles can be used together safely (probably more)
        table->table[j*width+i] = (exp(-(d)/(std*std)) / norm_max) * HEIGHT_SCALE;        
    }

}


void set_cauchy_table(lq_table *table, int std)
{
int i, j, x, y;
double d;
double norm_max;
int width, height;
width = table->width;
height = table->height;
//scale maximum height...
norm_max = 1.0/(std*8);
//set min
table->falloff_distance = min(table->falloff_distance, std*2);
for(i=0;i<width;i++)
    for(j=0;j<height;j++)
    {
        x = i - width/2;
        y = j - height/2;
        d = x*x + y*y;
        //Heights are scaled to max 2^23ish, so up to 512 particles can be used together safely (probably more)
        table->table[j*width+i] =  ((1.0/(d+std))/norm_max) * HEIGHT_SCALE;        
        //printf("%d\n", table->table[j*width+i]);
    }
}

void lut_set_pos(lq_lut *lut, int x, int y)
{
    lut->x_offset = x;
    lut->y_offset = y;
}

void lut_get_pos(lq_lut *lut, int *x, int *y)
{
    *x = lut->x_offset;
    *y = lut->y_offset;
}

lq_lut *new_lut(int x, int y, lq_table *table)
{
    lq_lut *lut;
    lut = calloc(sizeof(*lut), 1);
    lut->x_offset = x;
    lut->y_offset = y;
    lut->prev = NULL;
    lut->next = NULL;
    lut->blob_class = 0;
    lut->table = table;    
    return lut;
}

lq_data *new_data(int width, int height, int top, int bottom, int left, int right)
{
    int i;
    lq_data *data;
    data = calloc(sizeof(*data), 1);
    
    data->width = width;
    data->height = height;
    data->top = top;
    data->bottom = bottom;
    data->left = left;
    data->right = right;    
    data->edge_cache = calloc(sizeof(*(data->edge_cache)), width*height);
    data->y_edges = calloc(sizeof(*(data->y_edges)), width);
    for(i=0;i<width;i++)
        data->y_edges[i] = calloc(sizeof(*(data->y_edges[i])), MAX_EDGES);
    data->n_y_edges = calloc(sizeof(*(data->n_y_edges)), width);
    
    data->blob_sections = calloc(sizeof(*(data->blob_sections)), MAX_CLASSES);
    data->dir_cache = calloc(sizeof(*(data->dir_cache)), width*height);   
    data->class_cache = calloc(sizeof(*(data->class_cache)), width*height);   
    data->x_list = calloc(sizeof(*(data->x_list)), MAX_PTS);
    data->y_list = calloc(sizeof(*(data->y_list)), MAX_PTS);
    data->n_list_pts = 0;
    data->n_sections = 0;
    data->lut_list = NULL;    
    data->boundary = NULL;
    return data;
}

//boundary must be an array with w*h elements
void data_set_boundary(lq_data *data, int *boundary)
{
    data->boundary = boundary;
}

void data_add_lut(lq_data *data, lq_lut *lut)
{
    if(data->lut_list==NULL)
    {
        data->lut_list = lut;
        data->lut_list->next = NULL;
        data->lut_list->prev = NULL;        
     }
     else
     {
        lut->next = data->lut_list;
        data->lut_list->prev = lut;
        data->lut_list = lut;     
     }   
}

int data_remove_lut(lq_data *data, lq_lut *lut)
{
    lq_lut *lut_test;
    
    lut_test = data->lut_list;
    while(lut_test!=NULL)
    {
        if(lut_test == lut)
        {
            lut_test->prev->next = lut_test->next;
            lut_test->next->prev = lut_test->prev;
            return 1;
        }
        
        lut_test = lut_test->next;
    }
    return 0;        
}




int get_height(lq_data *data, int x, int y)
{
    lq_lut *lut;
    int max_dist;
    int height;
    int tx, ty, xoff, yoff;
    lut = data->lut_list;
    height = 0;
    while(lut!=NULL)
    {
        max_dist = lut->table->falloff_distance;
        tx = lut->x_offset;
        ty = lut->y_offset;
        //cutoff-check (also checks if we're inside the width/height of this table)
        if(!(abs(x-tx)>max_dist || abs(y-ty)>max_dist))
            {        
                //Work out the index
                xoff = tx - x + lut->table->width/2;
                yoff = ty - y + lut->table->height/2;
                //add height
                height += lut->table->table[xoff+yoff*lut->table->width];
            }
        lut = lut->next;    
    }
    
    return height;
}



int trace_pixel(lq_data *data, int x, int y, int thresh)
{
    int co;
    int h;
    
    //If outside cacheable area, always return an "under" value
    if(x<0 || x>=data->width || y<0 || y>=data->height || x<=data->left || x>=data->right || y<=data->top || y>=data->bottom)
    {
        return PIXEL_UNDER;    
    }
    
    
    co = x+y*data->width;
    
    //boundary test
    if(data->boundary)
    {
        if(data->boundary[co]>0.0)
            return PIXEL_UNDER;
    }
    
    
    
    
    // Check if we hit an edge already found, or we already know the thresholded value
    if(data->edge_cache[co]!=0)
        return data->edge_cache[co];
    
    //compute the height
    h = get_height(data, x, y);
    if(h>thresh)
        {
            data->edge_cache[co] = PIXEL_OVER;
            return PIXEL_OVER;
        }
        else
        {
            data->edge_cache[co]=PIXEL_UNDER;
            return PIXEL_UNDER;
        }      
}


//Find the direction to follow, return DIR_EDGE if an edge hit. 
int compute_direction(lq_data *data, int x, int y, int thresh, int *idx)
{
    int a;
    int index;
    index = 0;
    
    //CHANGED
    if(x<0 )
        return DIR_DOWN;
    if(y<0 )
        return DIR_LEFT;
    if(x>data->width-1)
        return DIR_UP;
    if(y>data->height-1)
        return DIR_RIGHT;
        
        
    a = trace_pixel(data,x,y,thresh);
    if(a==PIXEL_EDGE)
        return DIR_EDGE;
    else 
        index |= (a==PIXEL_OVER) ? 1 : 0;
    a = trace_pixel(data,x+1,y,thresh);
    if(a==PIXEL_EDGE)
        return DIR_EDGE;
    else 
        index |= (a==PIXEL_OVER) ? 2 : 0;
    a = trace_pixel(data,x,y+1,thresh);
    if(a==PIXEL_EDGE)
        return DIR_EDGE;
    else 
        index |= (a==PIXEL_OVER) ? 4 : 0;
        
    a = trace_pixel(data,x+1,y+1,thresh);
    if(a==PIXEL_EDGE)
        return DIR_EDGE;
    else 
        index |= (a==PIXEL_OVER) ? 8 : 0;
    *idx = index;
    if(index==0)
        {
            return DIR_EDGE;
        }
    return walking_table[index];
}

// Trace the isocontour
void edge_trace(lq_data *data, lq_lut *lut, int x, int y, int thresh)
{
    int dir, idx;
    int orig_x, orig_y;
    int ctr;
    orig_x = x;
    orig_y = y;
    ctr = 0;
    do 
    {
        dir = compute_direction(data, x,y, thresh, &idx);
        // This point is _always_ an edge
        if(dir!=DIR_EDGE)
            add_edge_point(data, x, y, idx);                
        switch(dir)
        {       
        case DIR_EDGE:
             // set the class this blob belongs to
             if(x>=0 && y>=0 && x<data->width && y<data->height)
                lut->blob_class = data->class_cache[x+y*data->width];
             return; break;        
        case DIR_BAD:
             
            return; break;
        case DIR_UP:
       
            y--; break;
        case DIR_DOWN:
       
            y++; break;
        case DIR_LEFT:
        
            x--; break;
        case DIR_RIGHT:
    
            x++; break;            
        }   
        ctr++;
    } while((x!=orig_x || y!=orig_y) && ctr<4000);
}

void add_edge_point(lq_data *data, int x, int y, int dir)
{

    if(x<0)
        x = 0;
     if(y<0)
        y = 0;
    if(y>=data->height)
        y = data->height-1;
    if(x>=data->width)
        x = data->width-1;
    
    
 
    data->dir_cache[x+y*data->width] = dir;
    data->x_list[data->n_list_pts] = x;
    data->y_list[data->n_list_pts] = y;
    if(data->n_list_pts<MAX_PTS)
        data->n_list_pts++;    
}



void trace(lq_data *data, lq_lut *lut, int thresh)
{
int x, y, res;

/* start at the centre */
x = lut->x_offset;
y = lut->y_offset;
    
/* don't trace off the screen */
if(x<0 || y<0 || x>=data->width || y>=data->height)
    return;

/* find the left hand edge */
while(x>=0)
{    
    res=trace_pixel(data, x,y,thresh);
    if(res==PIXEL_EDGE)
    {   
        lut->blob_class = data->class_cache[x+y*data->width];
        return;
    }
    if(res==PIXEL_UNDER)
        break;
    x--; // Move leftward
}

add_edge_point(data,x-1,y,0);

//Follow the edge
edge_trace(data, lut, x, y, thresh);
}

void set_edge(lq_data *data, int x, int y, int class)
{
    int co = x+y*data->width;
    //Set an edge, and update the class, if it's not done already
    if(data->edge_cache[co] != PIXEL_EDGE)
    {
        data->edge_cache[co] = PIXEL_EDGE;
        data->class_cache[co] = class; 
        
    }
    
}



void compute_vstrips(lq_data *data)
{
int x, y, i, j, k,  n, t, dir;
lq_edge *edge, *test_edge, *last_edge;
for(i=0;i<data->n_list_pts;i++)
    {
        x = data->x_list[i];
        y = data->y_list[i];
        dir = data->dir_cache[x+y*data->width];
        if(1)
        {
            n = data->n_y_edges[x];
            //increment number of edges
            data->n_y_edges[x]++;                                    
            
            //find insertion point
            j = 0;
            while(j<n && data->y_edges[x][j]<=y)
                j++;
            //shift along 1
            k = data->n_y_edges[x];
            while(k>j)
            {
                data->y_edges[x][k] = data->y_edges[x][k-1];
                k--;
            }
            
            
            //put in new value
            data->y_edges[x][j] = y;                       
        }
  }
}

void data_render(lq_data *data, int thresh)
{    
    lq_lut *lut_test;
    int i, k, blob_class;
    lut_test = data->lut_list;
    data->n_list_pts = 0;
    data->n_sections = 0;
    memset(data->edge_cache,0,data->width*data->height*sizeof(*(data->edge_cache)));
    memset(data->n_y_edges,0,data->width*sizeof(*(data->n_y_edges)));
    
    //???
    memset(data->class_cache,0,data->width*data->height*sizeof(*(data->class_cache)));
    memset(data->dir_cache,0,data->width*data->height*sizeof(*(data->dir_cache))); 
    
    blob_class = 0;
    while(lut_test!=NULL)
    {    
        lut_test->blob_class = blob_class;
        k = data->n_list_pts;
        trace(data, lut_test, thresh);
        
        //is this a new blob?
        if(lut_test->blob_class == blob_class)
        {
            data->blob_sections[blob_class].start = k;
            data->blob_sections[blob_class].end = data->n_list_pts;
        }
        else
        {
            //no blob
            data->blob_sections[blob_class].start = -1;
            data->blob_sections[blob_class].end = -1;
        }
        
        //mark all new data points
        for(i=k;i<data->n_list_pts;i++)        
            set_edge(data, data->x_list[i], data->y_list[i], blob_class);
        
        blob_class++;
        data->n_sections++;
        lut_test = lut_test->next;     
    }
   // compute_vstrips(data);
    

}