#include "ops.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Regular_triangulation_cell_base_3.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Cartesian_converter.h>
#include <chrono>
#include <memory>

// typedef CGAL::Exact_predicates_exact_constructions_kernel        K;      // change to this for exact computation
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel         K2;
typedef CGAL::Cartesian_converter<K, K2>                            ExactToFloat;

typedef CGAL::Regular_triangulation_vertex_base_3<K>                Vb0;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K, Vb0>    Vb;
typedef CGAL::Regular_triangulation_cell_base_3<K>                  Cb;

typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                        Tds;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb, CGAL::Parallel_tag>    pTds;

typedef CGAL::Regular_triangulation_3<K, Tds>                Triangulation;
typedef CGAL::Regular_triangulation_3<K, pTds>               pTriangulation;

typedef K::FT                                               Weight;
typedef K::Point_3                                          Point;
typedef K::Weighted_point_3                                 Weighted_point;

namespace CGALDDT {
    DTResult WDT(int num_points, int dimension,
                const float* positions,
                const float* weights,
                const bool weighted,
                const bool parallelize,
                const int p_lock_grid_size,
                const bool compute_cc) {
        
        assert(weighted);
        assert(dimension == 3);     // @TODO: support 2D

        // @TODO: ugly initialization, faster way?
        std::vector<std::pair<Weighted_point, Triangulation::Vertex::Info>> weighted_points;
        weighted_points.reserve(num_points);
        float inf = std::numeric_limits<float>::infinity();
        CGAL::Bbox_3 bbox(inf, inf, inf, -inf, -inf, -inf);
            
        for (int i = 0; i < num_points; i++) {
            Point p(
                positions[i * dimension + 0], 
                positions[i * dimension + 1], 
                positions[i * dimension + 2]);
            Weight w(weights[i]);
            weighted_points.push_back(std::make_pair(Weighted_point(p, w), i));
            bbox = bbox + p.bbox();
        }

        DTResult result;
        ExactToFloat exact_to_float;
        // @TODO: Fix this ugly code, redundant...
        if (parallelize) {

            auto triangulation_start_time = std::chrono::high_resolution_clock::now();
            pTriangulation triangulator;
            pTriangulation::Lock_data_structure locking_ds(bbox, p_lock_grid_size);
            triangulator.set_lock_data_structure(&locking_ds);
            triangulator.insert(weighted_points.begin(), weighted_points.end());

            auto triangulation_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> triangulation_time = (triangulation_end_time - triangulation_start_time);
            result.time_sec = triangulation_time.count();

            assert(triangulator.is_valid());
            assert(triangulator.dimension() == 3);

            result.num_tri = triangulator.number_of_finite_cells();
            result.tri_verts_idx = new int[result.num_tri * 4];
            if (compute_cc)
                result.tri_cc = new float[result.num_tri * 3];

            // @TODO: Faster way?
            int cnt = 0;
            int cnt2 = 0;
            for (auto it = triangulator.finite_cells_begin(); 
                    it != triangulator.finite_cells_end(); 
                    it++) {
                for (int i = 0; i < 4; i++)
                    result.tri_verts_idx[cnt++] = it->vertex(i)->info();
                if (compute_cc) {
                    auto cc = triangulator.dual(it);
                    result.tri_cc[cnt2 * 3 + 0] = exact_to_float(cc.x());
                    result.tri_cc[cnt2 * 3 + 1] = exact_to_float(cc.y());
                    result.tri_cc[cnt2 * 3 + 2] = exact_to_float(cc.z());
                    cnt2++;
                }
            }
        }
        else {
            auto triangulation_start_time = std::chrono::high_resolution_clock::now();

            Triangulation triangulator;
            triangulator.insert(weighted_points.begin(), weighted_points.end());

            auto triangulation_end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> triangulation_time = (triangulation_end_time - triangulation_start_time);
            result.time_sec = triangulation_time.count();
            
            assert(triangulator.is_valid());
            assert(triangulator.dimension() == 3);

            result.num_tri = triangulator.number_of_finite_cells();
            result.tri_verts_idx = new int[result.num_tri * 4];
            if (compute_cc)
                result.tri_cc = new float[result.num_tri * 3];

            // TODO: Faster way?
            int cnt = 0;
            int cnt2 = 0;
            for (auto it = triangulator.finite_cells_begin(); 
                    it != triangulator.finite_cells_end(); 
                    it++) {
                for (int i = 0; i < 4; i++)
                    result.tri_verts_idx[cnt++] = it->vertex(i)->info();

                if (compute_cc) {
                    auto cc = triangulator.dual(it);
                    result.tri_cc[cnt2 * 3 + 0] = exact_to_float(cc.x());
                    result.tri_cc[cnt2 * 3 + 1] = exact_to_float(cc.y());
                    result.tri_cc[cnt2 * 3 + 2] = exact_to_float(cc.z());
                    cnt2++;
                }
            }
        }
        
        return result;
    }
}