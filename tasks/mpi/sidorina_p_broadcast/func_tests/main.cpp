#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/sidorina_p_broadcast/include/ops_mpi.hpp"
#include "mpi/sidorina_p_broadcast/include/ops_mpi_m.hpp"

TEST(sidorina_p_broadcast_mpi, Test_broadcast_1) {
  boost::mpi::communicator world;

  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;
  std::vector<int> reference_res;

  std::shared_ptr<ppc::core::TaskData> taskDataGlob = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataRef = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_input = std::vector<int>({1, 2, 3});
    global_powers = std::vector<int>({1, 2});

    global_res.resize(global_input.size(), 0);

    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataGlob->inputs_count.emplace_back(global_input.size());
    taskDataGlob->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataGlob->inputs_count.emplace_back(global_powers.size());
    taskDataGlob->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataGlob->outputs_count.emplace_back(global_res.size());
  }

  sidorina_p_broadcast_mpi::Broadcast testMpiTaskParallel(taskDataGlob);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    reference_res.resize(global_input.size(), 0);

    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataRef->inputs_count.emplace_back(global_input.size());
    taskDataRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataRef->inputs_count.emplace_back(global_powers.size());
    taskDataRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataRef->outputs_count.emplace_back(reference_res.size());
  }
  sidorina_p_broadcast_mpi::RefBroadcast testMpiTaskSequential(taskDataRef);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  if (world.rank() == 0) {
    for (int i : global_res) std::cout << i;
    for (int i : reference_res) std::cout << i;
    ASSERT_EQ(global_res, reference_res);
  }
}
