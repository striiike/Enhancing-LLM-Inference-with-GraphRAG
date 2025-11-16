"""
Analysis script for test results
Analyzes the comparison between Enhanced and Baseline
"""
import json
import sys

def analyze_results(filename='final_test_78_exemplars.json'):
    """Analyze test results and generate detailed report"""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {filename}")
        return
    
    print("=" * 100)
    print("DETAILED ANALYSIS OF TEST RESULTS")
    print("=" * 100)
    
    # Summary
    summary = results['summary']
    print(f"\n📋 TEST METADATA:")
    print(f"  Model: {results['metadata']['model']}")
    print(f"  Test Date: {results['metadata']['test_date']}")
    print(f"  Total Exemplars: {results['metadata']['total_exemplars']}")
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"  Valid Comparisons: {summary['valid_comparisons']}")
    print(f"  Schema Failures: {summary['schema_prune_failures']}")
    
    print(f"\n🏆 WINS:")
    print(f"  Enhanced:  {summary['enhanced']['wins']} ({summary['enhanced']['win_rate']})")
    print(f"  Baseline:  {summary['baseline']['wins']} ({summary['baseline']['win_rate']})")
    print(f"  Ties:      {summary['ties']} ({summary['tie_rate']})")
    
    print(f"\n📊 TOTAL RESULTS RETURNED:")
    print(f"  Enhanced:  {summary['enhanced']['total_results']}")
    print(f"  Baseline:  {summary['baseline']['total_results']}")
    
    print(f"\n❌ ERROR RATES:")
    print(f"  Enhanced:  {summary['enhanced']['errors']}/{results['metadata']['total_exemplars']} ({summary['enhanced']['error_rate']})")
    print(f"  Baseline:  {summary['baseline']['errors']}/{results['metadata']['total_exemplars']} ({summary['baseline']['error_rate']})")
    
    # Analyze Enhanced wins
    print(f"\n" + "=" * 100)
    print(f"ENHANCED WINS ({summary['enhanced']['wins']} cases)")
    print("=" * 100)
    
    enhanced_wins = [d for d in results['details'] if d['winner'] == 'enhanced']
    for item in enhanced_wins[:10]:  # Show first 10
        print(f"\n[{item['index']}] {item['question']}")
        print(f"  Expected: {item['expected']}")
        print(f"  Enhanced: {item['enhanced']['count']} ✓")
        print(f"  Baseline: {item['baseline']['count']} | Error: {item['baseline'].get('error', 'None')[:50] if item['baseline'].get('error') else 'None'}")
    
    if len(enhanced_wins) > 10:
        print(f"\n... and {len(enhanced_wins) - 10} more Enhanced wins")
    
    # Analyze Baseline wins
    print(f"\n" + "=" * 100)
    print(f"BASELINE WINS ({summary['baseline']['wins']} cases)")
    print("=" * 100)
    
    baseline_wins = [d for d in results['details'] if d['winner'] == 'baseline']
    for item in baseline_wins:
        print(f"\n[{item['index']}] {item['question']}")
        print(f"  Expected: {item['expected']}")
        print(f"  Enhanced: {item['enhanced']['count']} | Error: {item['enhanced'].get('error', 'None')[:50] if item['enhanced'].get('error') else 'None'}")
        print(f"  Baseline: {item['baseline']['count']} ✓")
        if item['enhanced'].get('query'):
            print(f"  Enhanced Query: {item['enhanced']['query'][:100]}...")
        if item['baseline'].get('query'):
            print(f"  Baseline Query: {item['baseline']['query'][:100]}...")
    
    # Analyze ties where both got it right
    exact_ties = [d for d in results['details'] 
                  if d['winner'] == 'tie' 
                  and d['enhanced']['success'] 
                  and d['baseline']['success']
                  and d['enhanced']['count'] == d['expected']]
    
    print(f"\n" + "=" * 100)
    print(f"PERFECT TIES (both correct): {len(exact_ties)} cases")
    print("=" * 100)
    print(f"Both Enhanced and Baseline returned the exact expected count.\n")
    
    # Analyze cases where Enhanced got exact match
    enhanced_exact = [d for d in results['details'] 
                      if d['enhanced']['success'] 
                      and d['enhanced']['count'] == d['expected']]
    baseline_exact = [d for d in results['details'] 
                      if d['baseline']['success'] 
                      and d['baseline']['count'] == d['expected']]
    
    print(f"\n📍 EXACT MATCHES:")
    print(f"  Enhanced exact matches: {len(enhanced_exact)}/{results['metadata']['total_exemplars']} ({len(enhanced_exact)/results['metadata']['total_exemplars']*100:.1f}%)")
    print(f"  Baseline exact matches: {len(baseline_exact)}/{results['metadata']['total_exemplars']} ({len(baseline_exact)/results['metadata']['total_exemplars']*100:.1f}%)")
    
    # Final verdict
    print(f"\n" + "=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    
    if summary['enhanced']['wins'] > summary['baseline']['wins']:
        margin = summary['enhanced']['wins'] - summary['baseline']['wins']
        percentage = margin / summary['valid_comparisons'] * 100
        print(f"\n🎉 ENHANCED IS SUPERIOR!")
        print(f"  Winning margin: {margin} questions ({percentage:.1f}%)")
        print(f"  Enhanced dominance: {summary['enhanced']['wins']}/{summary['baseline']['wins']} win ratio")
    elif summary['baseline']['wins'] > summary['enhanced']['wins']:
        margin = summary['baseline']['wins'] - summary['enhanced']['wins']
        percentage = margin / summary['valid_comparisons'] * 100
        print(f"\n⚠️  BASELINE IS BETTER")
        print(f"  Winning margin: {margin} questions ({percentage:.1f}%)")
    else:
        print(f"\n🤝 TIED - No clear winner")
    
    print(f"\n" + "=" * 100)

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else 'final_test_78_exemplars.json'
    analyze_results(filename)
