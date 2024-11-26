#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sidorina_p_broadcast_mpi {

class Broadcast : public ppc::core::Task {
 public:
  explicit Broadcast(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  template <typename T>
  static void broadcast_m(const boost::mpi::communicator& comm, T& value, int root) {
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
  static void broadcast_m(const boost::mpi::communicator& comm, T* value, int n, int root) {
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
      recipients[(root + 1) % comm.size()] = *value;
      recipients[(root + 2) % comm.size()] = *value;
    } else {
      int id_el = comm.rank() - root;
      if (comm.rank() < root) {
        id_el = comm.size() - root + comm.rank();
      }

      int id_send = (root + (id_el - 1) / 2) % comm.size();

      comm.recv(id_send, 0, value, n);
    }
  }


  std::function<void(const boost::mpi::communicator&, int*, int, int)> broadcast_fn;

 private:
  std::vector<int> arr;
  std::vector<int> term;
  std::vector<int> res;
  int del = 0;
  int sz = 0;
  boost::mpi::communicator world;
};
}  // namespace sidorina_p_broadcast_mpi