__copyright__ = """ Copyright (c) 2010-2011 Torsten Schmits

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
"""

class Config(object):
    def __init__(self):
        self.bg_color = 'black'
        self.font_color_name = 'red'
        self.matrix_columns = 5
        self.nr_sequences = 3
        self.min_dist = 2
	self.stimulus_type = 1
        self.target_present_time = 1.
        self.highlight_time = .1
        self.between_highlights_time = .05
        # y position of the target phrase relative to top side
        self.word_vpos = 50
        self.word_font_size = 72
        self.word_target_font_size = 96
        # space above and below phrase
        self.word_margin = 10
        # display duration of the classifier input
        self.feedback_time = 1.
        # size of an unhighlighted group
        self.matrix_font_size = 72
        # relative size of a highlighted group
        self.magnification_factor = 1.2
        self._view_parameters.extend(['symbols', 'matrix_columns', 'word_vpos',
                                      'word_font_size', 'word_target_font_size',
                                      'word_margin', 'feedback_time',
                                      'magnification_factor',
                                      'matrix_font_size'])
