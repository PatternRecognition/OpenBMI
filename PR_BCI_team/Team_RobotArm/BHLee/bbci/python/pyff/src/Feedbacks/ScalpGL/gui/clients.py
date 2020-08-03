__copyright__ = """ Copyright (c) 2008, 2009 Torsten Schmits

This file is part of the pyff framework. pyff is free software;
you can redistribute it and/or modify it under the terms of the GNU General
Public License version 2, as published by the Free Software Foundation.

pyff is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA  02111-1307  USA

"""

import sip
from itertools import imap, ifilter, chain

from PyQt4.QtCore import qDebug
from PyQt4.QtGui import QWidget, QGridLayout

from .client import DummyClient

class Clients(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self.__init_attributes()
        self._reinit_dummies()
        self.setAcceptDrops(True)

    def __init_attributes(self):
        self.dummies = []
        self.clients = []
        self.config = { 'rows': 2, 'cols': 2 }

    def _reinit_dummies(self):
        self.clients = []
        old = self.layout()
        if old is not None:
            sip.delete(old)
        layout = QGridLayout(self)
        for r in range(0, self.config['rows']):
            for c in range(0, self.config['cols']):
                self.clients.append(DummyClient())
                layout.addWidget(self.clients[-1], r, c)

    def _reinit_dummies_old(self):
        for d in self.dummies:
            try:
                sip.delete(d)
            except:
                pass
        self.dummies = []
        old = self.layout()
        if old is not None:
            sip.delete(old)
        layout = QGridLayout(self)
        for r in range(0, self.config['rows']):
            for c in range(0, self.config['cols']):
                self.dummies.append(Client())
                layout.addWidget(self.dummies[-1], r, c)

    def insert_client_qt(self, client, index):
        index = self._qt_to_index(index)
        self.insert_client(client, index)

    def insert_client(self, client, index):
        row, column = self._index_to_coords(index)
        old_client = self.layout().itemAtPosition(row, column).widget()
        self.layout().addWidget(client, row, column)
        self.clients[index] = client
        sip.delete(old_client)

    def insert_client_old(self, client, index):
        row, column, ignore, ignore = self.layout().getItemPosition(index)
        print 'index, row, column:', index, row, column
        old_client = self.layout().itemAtPosition(row, column).widget()
        self.layout().addWidget(client, row, column)
        self.clients[index] = client
        try:
            self.dummies.remove(old_client)
        except ValueError:
            pass
        sip.delete(old_client)
        client.hide()
        client.show()

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def client_at(self, pos):
        child = self.childAt(pos)
        index = self.layout().indexOf(child)
        return child, index

    def update_layout(self):
        print 'new layout:', self.config['rows'], self.config['cols']
        #for client in self.clients:
            #self.layout().removeWidget(client)
        clients = self.clients
        self._reinit_dummies()
        index = 0
        dummies = filter(lambda c: c.type == DummyClient.type, clients)
        clients = filter(lambda c: c.type != DummyClient.type, clients)
        map(sip.delete, dummies + clients[self._max_clients:])
        clients = clients[:self._max_clients]
        map(self.insert_client, clients, range(len(clients)))
        self._balance_measures()

    def update_layout_old(self):
        print 'new layout:', self.config['rows'], self.config['cols']
        for client in self.clients.itervalues():
            self.layout().removeWidget(client)
            client.hide()
        self._reinit_dummies()
        old_clients = self.clients
        self.clients = {}
        cnt = self.layout().count()
        clients = old_clients.keys()
        reinsert = clients[:cnt]
        discard = clients[cnt:]
        print 'count', cnt
        print 'discard', discard
        popped = map(old_clients.pop, discard)
        print 'popped', popped
        map(sip.delete, popped)
        clients = old_clients.values()
        #self.clients.clear()
        print 'clients', clients
        for index, client in enumerate(clients):
            self.insert_client(client, index)
            client.show()
        self._balance_measures()

    def test(self):
        qDebug(str(self.clients))

    def _balance_measures(self):
        for c in range(self.config['cols']):
            self.layout().setColumnStretch(c, 1)
        for r in range(self.config['rows']):
            self.layout().setRowStretch(r, 1)

    def _index_to_coords(self, index):
        return divmod(index, self.config['cols'])

    def _coords_to_index(self, row, col):
        return row * self.config['cols'] + col

    def _qt_to_index(self, index):
        row, column, ignore, ignore = self.layout().getItemPosition(index)
        return self._coords_to_index(row, column)

    @property
    def _max_clients(self):
        return self.config['rows'] * self.config['cols']
