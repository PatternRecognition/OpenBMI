/*
 * buffer.h
 *
 * This is the header file for buffer.c. It contains the declarations for 
 * the functions in buffer.c and hides the static variables in buffer.c
 *
 *  2009/01/16 - Max Sagebaum
 *                - file created
 */

#ifndef BUFFER_H
#define BUFFER_H

#include "buffer.c"

/*********************************************
 *
 * Creates the arrays for the buffer. Sets the reading and writing
 * positions.
 * INPUT: lines         - The number of lines for the buffer
 *        tracks        - The number of tracks for the buffer
 *        sizePertrack  - The number of values for each track
 * OUTPUT: -
 * RETURN: 1 if the buffer was created, false otherwise
 *
 **********************************************/
static int  bufferCreate(int lines, int tracks, int sizePerTrack);

/*********************************************
 *
 * Writes the values in data into the given track.
 * INPUT: data          - The values which will be written to the track.
 *        dataLines     - The number of lines in data.
 *        writeTrack    - Specifies the track for the data.
 * OUTPUT: -
 *
 **********************************************/
static void bufferWrite(float* data, int dataLines, int writeChunk);

/*********************************************
 *
 * Gets the position in the buffer were the new data begins and the size of
 * new data.
 * INPUT: -
 * OUTPUT: dataStart    - The position where the new data begins
 *         dataSize     - The size of the new data
 *
 **********************************************/
static void bufferRead(double** dataStart, int* dataSize);

/*********************************************
 *
 * Deletes all arrays which the buffer uses.
 *
 **********************************************/
static void bufferClose();

#endif