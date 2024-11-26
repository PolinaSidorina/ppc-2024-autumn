#include "mpi/sidorina_p_broadcast/include/ops_mpi_m.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace boost::mpi;

template <typename T>
inline void sidorina_p_broadcast_mpi::Broadcast::broadcast_m(const boost::mpi::communicator& comm, T& value, int root) {
  int n = comm.size();
  if (n <= 2) {
    if (comm.rank() == root) {
      if (n == 1) {
        return;
      } else {
        comm.send(1 - root, 0, value);
      }
    } else {
      comm.recv(root, 0, value);
    }
    return;
  }

  std::vector<int> recipients(comm.size());

  if (comm.rank() == root) {
    recipients[(root + 1) % comm.size()] = value;
    recipients[(root + 2) % comm.size()] = value;
  } else {
    int id_elem = comm.rank() - root;
    if (comm.rank() < root) {
      id_elem = comm.size() - root + comm.rank();
    }

    int id_sender = (root + (id_elem - 1) / 2) % comm.size();
    comm.recv(id_sender, 0, value);
  }
}

template <typename T>
inline void sidorina_p_broadcast_mpi::Broadcast::broadcast_m(const boost::mpi::communicator& comm, T& value, int n,
                                                             int root) {
  if (n <= 2) {
    if (comm.rank() == root) {
      if (n == 1) {
        return;
      } else {
        comm.send(1 - root, 0, value, n);
      }
    } else {
      comm.recv(root, 0, value, n);
    }
    return;
  }

  std::vector<int> recipients(comm.size());

  if (comm.rank() == root) {
    recipients[(root + 1) % comm.size()] = value;
    recipients[(root + 2) % comm.size()] = value;
  } else {
    int id_elem = comm.rank() - root;
    if (comm.rank() < root) {
      id_elem = comm.size() - root + comm.rank();
    }

    int id_sender = (root + (id_elem - 1) / 2) % comm.size();

    comm.recv(id_sender, 0, value, n);
  }
}

bool sidorina_p_broadcast_mpi::Broadcast::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    int sz1 = taskData->inputs_count[0];
    int sz2 = taskData->inputs_count[1];
    del = sz2 / world.size() + (sz2 % world.size());
    sz = sz1;

    arr.assign(reinterpret_cast<const int*>(taskData->inputs[0]),
               reinterpret_cast<const int*>(taskData->inputs[0]) + sz1);
    term.assign(reinterpret_cast<const int*>(taskData->inputs[1]),
                reinterpret_cast<const int*>(taskData->inputs[1]) + sz2);
  }

  return true;
}

bool sidorina_p_broadcast_mpi::Broadcast::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] > 0 &&
           taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool sidorina_p_broadcast_mpi::Broadcast::run() {
  internal_order_test();

  int root = 0;
  broadcast(world, del, 0);
  broadcast(world, sz, 0);

  res.resize(sz, 0);
  if (world.rank() != root) {
    arr.resize(sz);
  } else {
    arr.resize(sz);
    std::copy(term.data(), term.data(), arr.begin());
  }

  broadcast(world, arr.data(), arr.size(), 0);

  if (world.rank() == root) {
    for (int p = 1; p < world.size(); ++p) {
      world.send(p, 0, term.data() + p * del, del);
    }
  } else {
    world.recv(0, 0, term.data(), del);
  }

  for (int i = 0; i < static_cast<int>(arr.size()); i++) {
    int num = arr[i];
    int result = 0;
    for (int t : term) {
      if (t >= 0) {
        result += num + t;
      }
    }
    arr[i] = result;
  }

  reduce(world, arr.data(), arr.size(), res.data(), std::plus<int>(), 0);

  return true;
}

bool sidorina_p_broadcast_mpi::Broadcast::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* answer = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(res.begin(), res.end(), answer);
  }
  return true;
}